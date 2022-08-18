# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# A shared LSTM + MAN for learning language-invariant features (F_s)
# Another shared LSTM with MoE on top for learning language-specific features (F_p)
# MoE MLP tagger (C)
from collections import defaultdict
import io
import itertools
import logging
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #Select your GPUs
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
import pickle
import random
import shutil
import sys
from tqdm import tqdm
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from options import opt
random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(opt.random_seed)
torch.use_deterministic_algorithms(True)

from data_prep.bio_dataset import *
from data_prep.eye_tracking import *
from models import *
import utils
from vocab import Vocab, TagVocab
import wandb
wandb.init(project="Multilingual-Model-Transfer", entity="{specify}", name=opt.model_save_file)

# save models and logging
if not os.path.exists(opt.model_save_file):
    os.makedirs(opt.model_save_file)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
log.addHandler(fh)
# output options
log.info(opt)


def train(vocabs, char_vocab, tag_vocab, train_sets, dev_sets, test_sets, unlabeled_sets, meco, numLoss):
    """
    train_sets, dev_sets, test_sets: dict[lang] -> AmazonDataset
    For unlabeled langs, no train_sets are available
    """
    # dataset loaders
    train_loaders, unlabeled_loaders = {}, {}
    train_iters, unlabeled_iters, d_unlabeled_iters = {}, {}, {}
    dev_loaders, test_loaders = {}, {}
    meco_loaders , meco_iters = {}, {}
    my_collate = utils.sorted_collate if opt.model=='lstm' else utils.unsorted_collate
    for lang in opt.langs:
        train_loaders[lang] = DataLoader(train_sets[lang],
                opt.batch_size, shuffle=True, collate_fn = my_collate, num_workers=0)
        train_iters[lang] = iter(train_loaders[lang])
    for lang in opt.dev_langs:
        dev_loaders[lang] = DataLoader(dev_sets[lang],
                opt.batch_size, shuffle=False, collate_fn = my_collate, num_workers=0)
        test_loaders[lang] = DataLoader(test_sets[lang],
                opt.batch_size, shuffle=False, collate_fn = my_collate, num_workers=0)
    for lang in opt.all_langs:
        if lang in opt.unlabeled_langs:
            uset = unlabeled_sets[lang]
        else:
            # for labeled langs, consider which data to use as unlabeled set
            if opt.unlabeled_data == 'both':
                uset = ConcatDataset([train_sets[lang], unlabeled_sets[lang]])
            elif opt.unlabeled_data == 'unlabeled':
                uset = unlabeled_sets[lang]
            elif opt.unlabeled_data == 'train':
                uset = train_sets[lang]
            else:
                raise Exception(f'Unknown options for the unlabeled data usage: {opt.unlabeled_data}')
        unlabeled_loaders[lang] = DataLoader(uset,
                opt.batch_size, shuffle=True, collate_fn = my_collate, num_workers=0)
        unlabeled_iters[lang] = iter(unlabeled_loaders[lang])
        d_unlabeled_iters[lang] = iter(unlabeled_loaders[lang])
        meco_loaders[lang] = DataLoader(meco[lang],
                    opt.batch_size, shuffle=True, collate_fn = utils.sorted_collate_gaze, num_workers=0)
        meco_iters[lang] = iter(meco_loaders[lang])

    # embeddings
    emb = MultiLangWordEmb(vocabs, char_vocab, opt.use_wordemb, opt.use_charemb).to(opt.device)
    # models
    F_s = None
    F_p = None
    C, D = None, None
    num_experts = len(opt.langs)+1 if opt.expert_sp else len(opt.langs)
    if opt.model.lower() == 'lstm':
        if opt.shared_hidden_size > 0:
            F_s = LSTMFeatureExtractor(opt.total_emb_size, opt.F_layers, opt.shared_hidden_size,
                                       opt.word_dropout, opt.dropout, opt.bdrnn)
        if opt.private_hidden_size > 0:
            if not opt.concat_sp:
                assert opt.shared_hidden_size == opt.private_hidden_size, "shared dim != private dim when using add_sp!"
            F_p = nn.Sequential(
                    LSTMFeatureExtractor(opt.total_emb_size, opt.F_layers, opt.private_hidden_size,
                            opt.word_dropout, opt.dropout, opt.bdrnn),
                    MixtureOfExperts(opt.MoE_layers, opt.private_hidden_size,
                            len(opt.langs), opt.private_hidden_size,
                            opt.private_hidden_size, opt.dropout, opt.MoE_bn, False)
                    )
    else:
        raise Exception(f'Unknown model architecture {opt.model}')

    if opt.C_MoE:
        C = SpMixtureOfExperts(opt.C_layers, opt.shared_hidden_size, opt.private_hidden_size, opt.concat_sp,
                num_experts, opt.shared_hidden_size + opt.private_hidden_size, len(tag_vocab),
                opt.mlp_dropout, opt.C_bn)
    else:
        C = SpMlpTagger(opt.C_layers, opt.shared_hidden_size, opt.private_hidden_size, opt.concat_sp,
                opt.shared_hidden_size + opt.private_hidden_size, len(tag_vocab),
                opt.mlp_dropout, opt.C_bn)
    if opt.shared_hidden_size > 0 and opt.n_critic > 0:
        if opt.D_model.lower() == 'lstm':
            d_args = {
                'num_layers': opt.D_lstm_layers,
                'input_size': opt.shared_hidden_size,
                'hidden_size': opt.shared_hidden_size,
                'word_dropout': opt.D_word_dropout,
                'dropout': opt.D_dropout,
                'bdrnn': opt.D_bdrnn,
                'attn_type': opt.D_attn
            }
        elif opt.D_model.lower() == 'cnn':
            d_args = {
                'num_layers': 1,
                'input_size': opt.shared_hidden_size,
                'hidden_size': opt.shared_hidden_size,
                'kernel_num': opt.D_kernel_num,
                'kernel_sizes': opt.D_kernel_sizes,
                'word_dropout': opt.D_word_dropout,
                'dropout': opt.D_dropout
            }
        else:
            d_args = None

        if opt.D_model.lower() == 'mlp':
            D = MLPLanguageDiscriminator(opt.D_layers, opt.shared_hidden_size,
                    opt.shared_hidden_size, len(opt.all_langs), opt.loss, opt.D_dropout, opt.D_bn)
        else:
            if opt.use_mt_learning:
                D = LanguageDiscriminatorMT(opt.D_model, opt.D_layers,
                        opt.shared_hidden_size, opt.shared_hidden_size,
                        len(opt.all_langs), opt.n_gaze_feat, opt.D_dropout, opt.D_bn, d_args)
            else:
                D = LanguageDiscriminator(opt.D_model, opt.D_layers,
                        opt.shared_hidden_size, opt.shared_hidden_size,
                        len(opt.all_langs), opt.D_dropout, opt.D_bn, d_args)
    if opt.use_data_parallel:
        F_s, C, D = nn.DataParallel(F_s).to(opt.device) if F_s else None, nn.DataParallel(C).to(opt.device), nn.DataParallel(D).to(opt.device) if D else None
    else:
        F_s, C, D = F_s.to(opt.device) if F_s else None, C.to(opt.device), D.to(opt.device) if D else None
    if F_p:
        if opt.use_data_parallel:
            F_p = nn.DataParallel(F_p).to(opt.device)
        else:
            F_p = F_p.to(opt.device)
    # optimizers
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(*map(list,
        [emb.parameters(), F_s.parameters() if F_s else [], \
        C.parameters(), F_p.parameters() if F_p else []]))),
        lr=opt.learning_rate,
        weight_decay=opt.weight_decay)
    if D:
        optimizerD = optim.Adam(D.parameters(), lr=opt.D_learning_rate, weight_decay=opt.D_weight_decay)

    # testing
    if opt.test_only:
        log.info(f'Loading model from {opt.model_save_file}...')
        if F_s:
            F_s.load_state_dict(torch.load(os.path.join(opt.model_save_file,
                f'netF_s.pth')))
        for lang in opt.all_langs:
            F_p.load_state_dict(torch.load(os.path.join(opt.model_save_file,
                f'net_F_p.pth')))
        C.load_state_dict(torch.load(os.path.join(opt.model_save_file,
            f'netC.pth')))
        if D:
            D.load_state_dict(torch.load(os.path.join(opt.model_save_file,
                f'netD.pth')))

        log.info('Evaluating validation sets:')
        acc = {}
        rec = {}
        prec = {}
        log.info(dev_loaders)
        log.info(vocabs)
        for lang in opt.dev_langs:
            acc[lang], rec[lang], prec[lang] = evaluate(f'{lang}_dev', dev_loaders[lang], vocabs[lang], tag_vocab,
                    emb, lang, opt.evalset, F_s, F_p, C)
        avg_acc = sum([acc[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        log.info(f'Average validation accuracy: {avg_acc}')
        log.info('Evaluating test sets:')
        wandb.log({f'valf1/f1-score {lang}':acc[lang], f'valrecall/recall {lang}':rec[lang], f'valprec/precision {lang}': prec[lang]})
        test_acc = {}
        test_rec = {}
        test_prec = {}
        for lang in opt.dev_langs:
            test_acc[lang], test_rec[lang], test_prec[lang] = evaluate(f'{lang}_test', test_loaders[lang],
                    vocabs[lang], tag_vocab, emb, lang, opt.evalset, F_s, F_p, C)
        avg_test_acc = sum([test_acc[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        log.info(f'Average test accuracy: {avg_test_acc}')
        wandb.log({f'testf1/f1-score {lang}':test_acc[lang], f'testrecall/recall {lang}':test_rec[lang], f'testprec/precision {lang}': test_prec[lang]})
        return {'valid': acc, 'test': test_acc}

    # training
    best_acc, best_avg_acc = defaultdict(float), 0.0
    best_rec, best_avg_rec = defaultdict(float), 0.0
    best_prec, best_avg_prec = defaultdict(float), 0.0
    epochs_since_decay = 0
    # lambda scheduling
    if opt.lambd > 0 and opt.lambd_schedule:
        opt.lambd_orig = opt.lambd
    num_iter = int(utils.gmean([len(train_loaders[l]) for l in opt.langs]))
    # adapt max_epoch
    if opt.max_epoch > 0 and num_iter * opt.max_epoch < 15000:
        opt.max_epoch = 15000 // num_iter
        log.info(f"Setting max_epoch to {opt.max_epoch}")
    for epoch in range(opt.max_epoch):
        emb.train()
        if F_s:
            F_s.train()
        C.train()
        if D:
            D.train()
        if F_p:
            F_p.train()
            
        # lambda scheduling
        if hasattr(opt, 'lambd_orig') and opt.lambd_schedule:
            if epoch == 0:
                opt.lambd = opt.lambd_orig
            elif epoch == 5:
                opt.lambd = 10 * opt.lambd_orig
            elif epoch == 15:
                opt.lambd = 100 * opt.lambd_orig
            log.info(f'Scheduling lambda = {opt.lambd}')

        # training accuracy
        correct, total = defaultdict(int), defaultdict(int)
        gate_correct = defaultdict(int)
        c_gate_correct = defaultdict(int)
        # D accuracy
        d_correct, d_total = 0, 0
        for k in tqdm(range(num_iter), ascii=True):
            # D iterations
            if opt.shared_hidden_size > 0:
                utils.freeze_net(emb)
                utils.freeze_net(F_s)
                utils.freeze_net(F_p)
                utils.freeze_net(C)
                utils.unfreeze_net(D)
                # WGAN n_critic trick since D trains slower
                n_critic = opt.n_critic
                if opt.wgan_trick:
                    if opt.n_critic>0 and ((epoch==0 and k<25) or k%500==0):
                        n_critic = 100

                for _ in range(n_critic):
                    D.zero_grad()
                    loss_d = {}
                    lang_features = {}
                    # train on both labeled and unlabeled langs
                    for lang in opt.all_langs:
                        for i in range(0,numLoss):
                            if i == 0:
                                # targets not used
                                d_inputs, _ = utils.endless_get_next_batch(
                                        unlabeled_loaders, d_unlabeled_iters, lang)
                            else:
                                d_inputs, d_targets_gaze =utils.endless_get_next_batch_gaze( meco_loaders,
                                        meco_iters, lang)
                            d_inputs, d_lengths, mask, d_chars, d_char_lengths = d_inputs
                            d_embeds = emb(lang, d_inputs, d_chars, d_char_lengths)
                            shared_feat = F_s((d_embeds, d_lengths))
                            if opt.grad_penalty != 'none':
                                lang_features[lang] = shared_feat.detach()
                            if opt.D_model.lower() == 'mlp':
                                d_outputs = D(shared_feat)
                                # if token-level D, we can reuse the gate label generator
                                d_targets = utils.get_gate_label(d_outputs, lang, mask, False, all_langs=True)
                                d_total += torch.sum(d_lengths).item()
                            else:
                                if i ==0:
                                    if opt.use_mt_learning:
                                        d_outputs, _ = D((shared_feat, d_lengths))
                                    else:
                                        d_outputs = D((shared_feat, d_lengths))
                                else:
                                    d_outputs, d_outputs1 = D((shared_feat, d_lengths))
                                d_targets = utils.get_lang_label(opt.loss, lang, len(d_lengths))
                                d_total += len(d_lengths)
                            if i==0:
                                # D accuracy
                                _, pred = torch.max(d_outputs, -1)
                                # d_total += len(d_lengths)
                                d_correct += (pred==d_targets).sum().item()
                                if opt.use_data_parallel:
                                    l_d = functional.nll_loss(d_outputs.view(-1, D.module.num_langs),
                                            d_targets.view(-1), ignore_index=-1)
                                else:
                                    l_d = functional.nll_loss(d_outputs.view(-1, D.num_langs),
                                            d_targets.view(-1), ignore_index=-1)
                                wandb.log({f"loss l_d D {lang}":l_d})
                            else:
                                l_d1 = functional.l1_loss(d_outputs1.view(-1),d_targets_gaze.view(-1))
                                l_d1 = opt.loss_multi * l_d1 + opt.loss_add
                                wandb.log({f"loss l_d1 D {lang}":l_d1})
                                weight = functional.softmax(torch.randn(2), dim=-1)
                                l_d = l_d*weight[0] + l_d1*weight[1]
                        wandb.log({f"loss D {lang}": l_d})
                        l_d.backward()
                        loss_d[lang] = l_d.item()
                    # gradient penalty
                    if opt.grad_penalty != 'none':
                        gp = utils.calc_gradient_penalty(D, lang_features,
                                onesided=opt.onesided_gp, interpolate=(opt.grad_penalty=='wgan'))
                        gp.backward()
                    optimizerD.step()

            # F&C iteration
            utils.unfreeze_net(emb)
            if opt.use_wordemb and opt.fix_emb:
                for lang in emb.langs:
                    emb.wordembs[lang].weight.requires_grad = False
            if opt.use_charemb and opt.fix_charemb:
                emb.charemb.weight.requires_grad = False
            utils.unfreeze_net(F_s)
            utils.unfreeze_net(F_p)
            utils.unfreeze_net(C)
            utils.freeze_net(D)
            emb.zero_grad()
            if F_s:
                F_s.zero_grad()
            if F_p:
                F_p.zero_grad()
            C.zero_grad()
            # optimizer.zero_grad()
            for lang in opt.langs:
                inputs, targets = utils.endless_get_next_batch(
                        train_loaders, train_iters, lang)
                inputs, lengths, mask, chars, char_lengths = inputs
                bs, seq_len = inputs.size()
                embeds = emb(lang, inputs, chars, char_lengths)
                shared_feat, private_feat = None, None
                if opt.shared_hidden_size > 0:
                    shared_feat = F_s((embeds, lengths))
                if opt.private_hidden_size > 0:
                    private_feat, gate_outputs = F_p((embeds, lengths))
                if opt.C_MoE:
                    c_outputs, c_gate_outputs = C((shared_feat, private_feat))
                else:
                    c_outputs = C((shared_feat, private_feat))
                # targets are padded with -1
                l_c = functional.nll_loss(c_outputs.view(bs*seq_len, -1),
                        targets.view(-1), ignore_index=-1)
                # gate loss
                if F_p:
                    gate_targets = utils.get_gate_label(gate_outputs, lang, mask, False)
                    l_gate = functional.cross_entropy(gate_outputs.view(bs*seq_len, -1),
                            gate_targets.view(-1), ignore_index=-1)
                    l_c += opt.gate_loss_weight * l_gate
                    _, gate_pred = torch.max(gate_outputs.view(bs*seq_len, -1), -1)
                    gate_correct[lang] += (gate_pred == gate_targets.view(-1)).sum().item()
                if opt.C_MoE and opt.C_gate_loss_weight > 0:
                    c_gate_targets = utils.get_gate_label(c_gate_outputs, lang, mask, opt.expert_sp)
                    _, c_gate_pred = torch.max(c_gate_outputs.view(bs*seq_len, -1), -1)
                    if opt.expert_sp:
                        l_c_gate = functional.binary_cross_entropy_with_logits(
                                mask.unsqueeze(-1) * c_gate_outputs, c_gate_targets)
                        c_gate_correct[lang] += torch.index_select(c_gate_targets.view(bs*seq_len, -1),
                                -1, c_gate_pred.view(bs*seq_len)).sum().item()
                    else:
                        l_c_gate = functional.cross_entropy(c_gate_outputs.view(bs*seq_len, -1),
                                c_gate_targets.view(-1), ignore_index=-1)
                        c_gate_correct[lang] += (c_gate_pred == c_gate_targets.view(-1)).sum().item()
                    l_c += opt.C_gate_loss_weight * l_c_gate
                l_c.backward()
                _, pred = torch.max(c_outputs, -1)
                total[lang] += torch.sum(lengths).item()
                correct[lang] += (pred == targets).sum().item()

            # update F with D gradients on all langs
            if D:
                for lang in opt.all_langs:
                    for i in range(0,numLoss):
                        if i==0:
                            inputs, _ = utils.endless_get_next_batch(
                                    unlabeled_loaders, unlabeled_iters, lang)
                        else:
                            inputs, d_targets_gaze =utils.endless_get_next_batch_gaze( meco_loaders,
                                    meco_iters, lang)
                        inputs, lengths, mask, chars, char_lengths = inputs
                        embeds = emb(lang, inputs, chars, char_lengths)
                        shared_feat = F_s((embeds, lengths))
                        # d_outputs = D((shared_feat, lengths))
                        if opt.D_model.lower() == 'mlp':
                            d_outputs, d_outputs1 = D(shared_feat)
                            # if token-level D, we can reuse the gate label generator
                            d_targets = utils.get_gate_label(d_outputs, lang, mask, False, all_langs=True)
                        else:
                            if i==0:
                                if opt.use_mt_learning:
                                    d_outputs, _  = D((shared_feat, lengths))
                                else:
                                    d_outputs = D((shared_feat, lengths))
                            else:
                                d_outputs, d_outputs1 = D((shared_feat, lengths))
                            d_targets = utils.get_lang_label(opt.loss, lang, len(lengths))
                        if i==0:
                            if opt.use_data_parallel:
                                l_d = functional.nll_loss(d_outputs.view(-1, D.module.num_langs),
                                        d_targets.view(-1), ignore_index=-1)
                            else:
                                l_d = functional.nll_loss(d_outputs.view(-1, D.num_langs),
                                        d_targets.view(-1), ignore_index=-1)
                            if opt.lambd > 0:
                                l_d *= -opt.lambd
                            wandb.log({f"loss l_d F {lang}": l_d})
                        else:
                            l_d1 = functional.l1_loss(d_outputs1.view(-1),d_targets_gaze.view(-1))
                            l_d1 = opt.loss_multi * l_d1 + opt.loss_add
                            if opt.lambd > 0:
                                l_d1 *= -opt.lambd
                            wandb.log({f"loss l_d1 F {lang}": l_d1})
                            weight = functional.softmax(torch.randn(2), dim=-1)
                            l_d = l_d*weight[0] + l_d1*weight[1]
                    wandb.log({f"loss F {lang}": l_d})
                    l_d.backward()

            optimizer.step()

        # end of epoch
        log.info('Ending epoch {}'.format(epoch+1))
        if d_total > 0:
            log.info('D Training Accuracy: {}%'.format(100.0*d_correct/d_total))
        log.info('Training accuracy:')
        log.info('\t'.join(opt.langs))
        log.info('\t'.join([str(100.0*correct[d]/total[d]) for d in opt.langs]))
        log.info('Gate accuracy:')
        log.info('\t'.join([str(100.0*gate_correct[d]/total[d]) for d in opt.langs]))
        log.info('Tagger Gate accuracy:')
        log.info('\t'.join([str(100.0*c_gate_correct[d]/total[d]) for d in opt.langs]))
        log.info('Evaluating validation sets:')
        acc = {}
        rec = {}
        prec = {}
        for lang in opt.dev_langs:
            acc[lang], rec[lang], prec[lang] = evaluate(f'{lang}_dev', dev_loaders[lang], vocabs[lang], tag_vocab,
                    emb, lang, opt.evalset, F_s, F_p, C)
        avg_acc = sum([acc[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        avg_rec = sum([rec[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        avg_prec = sum([prec[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        log.info(f'Average validation accuracy: {avg_acc}')
        log.info(f'Average validation recall: {avg_rec}')
        log.info(f'Average validation precision: {avg_prec}')
        wandb.log({f"Average validation accuracy {opt.unlabeled_langs}": avg_acc})
        wandb.log({f"Average validation recall {opt.unlabeled_langs}": avg_rec})
        wandb.log({f"Average validation precision {opt.unlabeled_langs}": avg_prec})

        log.info('Evaluating test sets:')
        test_acc = {}
        test_rec = {}
        test_prec = {}
        for lang in opt.dev_langs:
            test_acc[lang], test_rec[lang], test_prec[lang] = evaluate(f'{lang}_test', test_loaders[lang],
                    vocabs[lang], tag_vocab, emb, lang, opt.evalset, F_s, F_p, C)
        avg_test_acc = sum([test_acc[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        avg_test_rec = sum([test_rec[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        avg_test_prec = sum([test_prec[d] for d in opt.dev_langs]) / len(opt.dev_langs)
        log.info(f'Average test accuracy: {avg_test_acc}')
        log.info(f'Average test recall: {avg_test_rec}')
        log.info(f'Average test precision: {avg_test_prec}')
        wandb.log({f"Average test accuracy {opt.unlabeled_langs}": avg_test_acc})
        wandb.log({f"Average test recal {opt.unlabeled_langs}": avg_test_rec})
        wandb.log({f"Average test precision {opt.unlabeled_langs}": avg_test_prec})

        if avg_acc > best_avg_acc:
            epochs_since_decay = 0
            log.info(f'New best average validation accuracy: {avg_acc}')
            best_acc['valid'] = acc
            best_acc['test'] = test_acc
            best_rec['valid'] = rec
            best_rec['test'] = test_rec
            best_prec['valid'] = prec
            best_prec['test'] = test_prec
            best_avg_acc = avg_acc
            best_avg_rec = avg_rec
            best_avg_prec = avg_prec
            with open(os.path.join(opt.model_save_file, 'options.pkl'), 'wb') as ouf:
                pickle.dump(opt, ouf)
            if F_s:
                torch.save(F_s.state_dict(),
                        '{}/netF_s.pth'.format(opt.model_save_file))
            torch.save(emb.state_dict(),
                    '{}/net_emb.pth'.format(opt.model_save_file))
            if F_p:
                torch.save(F_p.state_dict(),
                        '{}/net_F_p.pth'.format(opt.model_save_file))
            torch.save(C.state_dict(),
                    '{}/netC.pth'.format(opt.model_save_file))
            if D:
                torch.save(D.state_dict(),
                        '{}/netD.pth'.format(opt.model_save_file))
        else:
            epochs_since_decay += 1
            if opt.lr_decay < 1 and epochs_since_decay >= opt.lr_decay_epochs:
                epochs_since_decay = 0
                old_lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = old_lr * opt.lr_decay
                log.info(f'Decreasing LR to {old_lr * opt.lr_decay}')

    # end of training
    log.info(f'Best average validation accuracy: {best_avg_acc}')
    log.info(f'Best average validation recal: {best_avg_rec}')
    log.info(f'Best average validation precision: {best_avg_prec}')
    return best_acc, best_rec, best_prec

def evaluate(name, loader, vocab, tag_vocab, emb, lang, evalset, F_s, F_p, C):
    emb.eval()
    if F_s:
        F_s.eval()
    if F_p:
        F_p.eval()
    C.eval()
    it = iter(loader)
    conll = io.StringIO()
    wikiann = []
    org_lab = []
    with torch.no_grad():
        for inputs, targets in tqdm(it, ascii=True):
            inputs, lengths, mask, chars, char_lengths = inputs
            embeds = (emb(lang, inputs, chars, char_lengths), lengths)
            shared_features, lang_features = None, None
            if opt.shared_hidden_size > 0:
                shared_features = F_s(embeds)
            if opt.private_hidden_size > 0:
                if not F_p:
                    # unlabeled lang
                    if opt.use_data_parallel:
                        lang_features = torch.zeros(target.size(0),
                                targets.size(1), opt.private_hidden_size)
                        lang_features = nn.DataParallel(lang_features).to(opt.device)
                    else:
                        lang_features = torch.zeros(targets.size(0),
                                targets.size(1), opt.private_hidden_size).to(opt.device)
                else:
                    lang_features, gate_outputs = F_p(embeds)
            if opt.C_MoE:
                outputs, _ = C((shared_features, lang_features))
            else:
                outputs = C((shared_features, lang_features))
            _, pred = torch.max(outputs, -1)
            bs, seq_len = pred.size()
            for i in range(bs):
                for j in range(lengths[i]):
                    word = vocab.get_word(inputs[i][j])
                    gold_tag = tag_vocab.get_tag(targets[i][j])
                    pred_tag = tag_vocab.get_tag(pred[i][j])
                    if evalset=='conll':
                        conll.write(f'{word} {gold_tag} {pred_tag}\n')
                    elif evalset=='wikiann':
                        wikiann.append(pred_tag)
                        org_lab.append(gold_tag)
                conll.write('\n')
    if evalset=='conll':
        f1, recall, precision = utils.conllF1(conll, log)
    elif evalset=='wikiann':
        f1, recall, precision = utils.wikiann(wikiann, org_lab)
    conll.close()
    log.info('{}: F1 score: {}%'.format(name, f1))
    log.info('{}: Recall: {}%'.format(name, recall))
    log.info('{}: Precision: {}%'.format(name, precision))
    return f1, recall, precision

def main():
    wandb.config.update(opt)
    if not os.path.exists(opt.model_save_file):
        os.makedirs(opt.model_save_file)
    if opt.use_mt_learning:
        print('Multi-Task learning Activated')
        numLoss = 2
    else:
        numLoss = 1
    log.info('Running the S-MAN + P-MoE + C-MoE model...')
    vocabs = {}
    tag_vocab = TagVocab()
    assert opt.use_wordemb or opt.use_charemb, "At least one of word or char embeddings must be used!"
    char_vocab = Vocab(opt.charemb_size)
    log.info(f'Loading Datasets...')
    log.info(f'Languages {opt.langs}')

    log.info('Loading Embeddings...')
    train_sets, dev_sets, test_sets, unlabeled_sets = {}, {}, {}, {}
    meco = {}
    for lang in opt.all_langs:
        log.info(f'Building Vocab for {lang}...')
        vocabs[lang] = Vocab(opt.emb_size, opt.emb_filenames[lang])
        assert not opt.train_on_translation or not opt.test_on_translation
        get_dataset_fn = get_ner_datasets
        if opt.train_on_translation:
            get_dataset_fn = get_train_on_translation_ner_datasets
        if opt.test_on_translation:
            get_dataset_fn = get_test_on_translation_ner_datasets
        train_sets[lang], dev_sets[lang], test_sets[lang], unlabeled_sets[lang] = \
                get_dataset_fn(vocabs[lang], char_vocab, tag_vocab, opt.ner_dir, lang)
        meco[lang] = get_gaze_data(vocabs[lang], char_vocab, opt.gaze_dir, 'meco', lang)

    opt.num_labels = len(tag_vocab)
    log.info(f'Tagset: {tag_vocab.id2tag}')
    log.info(f'Done Loading Datasets.')

    cv, cv1, cv2 = train(vocabs, char_vocab, tag_vocab, train_sets, dev_sets, test_sets, unlabeled_sets, meco, numLoss)
    log.info(f'Training done...')
    acc = sum(cv['valid'].values()) / len(cv['valid'])
    rec = sum(cv1['valid'].values()) / len(cv1['valid'])
    prec = sum(cv2['valid'].values()) / len(cv2['valid'])
    log.info(f'Validation Set Domain Average\t{acc}')
    log.info(f'Validation recall Set Domain Average\t{rec}')
    log.info(f'Validation precision Set Domain Average\t{prec}')
    test_acc = sum(cv['test'].values()) / len(cv['test'])
    test_rec = sum(cv1['test'].values()) / len(cv1['test'])
    test_prec = sum(cv2['test'].values()) / len(cv2['test'])
    log.info(f'Test Set Domain Average\t{test_acc}')
    log.info(f'Test recall Set Domain Average\t{test_rec}')
    log.info(f'Test precision Set Domain Average\t{test_prec}')
    return cv


if __name__ == '__main__':
    main()
