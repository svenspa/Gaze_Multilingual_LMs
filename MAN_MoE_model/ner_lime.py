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
os.environ["CUDA_VISIBLE_DEVICES"]="5" #Select your GPUs
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

import eli5
from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler

from optionsne import opt
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
wandb.init(project="LIME_NER", entity="svespa", name=opt.model_save_file)

# save models and logging
if not os.path.exists(opt.model_save_file):
    os.makedirs(opt.model_save_file)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
log.addHandler(fh)
# output options
log.info(opt)


def train(vocabs, char_vocab, tag_vocab, train_sets, dev_sets, test_sets, unlabeled_sets):
    """
    train_sets, dev_sets, test_sets: dict[lang] -> AmazonDataset
    For unlabeled langs, no train_sets are available
    """
    # dataset loaders
    train_loaders, unlabeled_loaders = {}, {}
    train_iters, unlabeled_iters, d_unlabeled_iters = {}, {}, {}
    dev_loaders, test_loaders = {}, {}
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

    log.info(f'Loading model from {opt.model_save_file}...')
    if F_s:
        F_s.load_state_dict(torch.load(os.path.join(opt.model_save_file,
            f'netF_s.pth')))
    for lang in opt.all_langs:
        F_p.load_state_dict(torch.load(os.path.join(opt.model_save_file,
            f'net_F_p.pth')))
    C.load_state_dict(torch.load(os.path.join(opt.model_save_file,
        f'netC.pth')))

    for lang in opt.dev_langs:
        evaluate(f'{lang}_dev', dev_loaders[lang], vocabs[lang], tag_vocab,
                emb, lang, opt.evalset, F_s, F_p, C)
    for lang in opt.dev_langs:
        evaluate(f'{lang}_test', test_loaders[lang], vocabs[lang], tag_vocab,
                emb, lang, opt.evalset, F_s, F_p, C)
    return None #Results are saved in WANDB, We do not need to return something

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
    right_pred = True
    with torch.no_grad():
        for inputs, targets in tqdm(it, ascii=True):
            inputs, lengths, mask, chars, char_lengths = inputs
            embeds = (emb(lang, inputs, chars, char_lengths), lengths)
            embeds
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
            outputs, _ = C((shared_features, lang_features))
            _, pred = torch.max(outputs, -1)
            bs, seq_len = pred.size()

            for i in range(bs):

                def get_pred_emb(inp):
                    ch = chars[i]
                    chl = char_lengths[i]
                    ch[ch!=0] = 0
                    ch = ch[0:lengths[i].item(),:]
                    chl[chl!=0] = 0
                    chl = chl[0:lengths[i].item()]
                    r1 = []
                    for k in range(len(inp)):
                        tup = inp[k].split()
                        r2 = []
                        for l in tup:
                            r2.append(vocab.lookup(l))
                        r1.append(r2)
                    return torch.tensor(r1).to(opt.device), torch.stack(len(inp)*[lengths[i]]).to(opt.device),torch.stack(len(inp)*[mask[i]]).to(opt.device),torch.stack(len(inp)*[ch]).to(opt.device), torch.stack(len(inp)*[chl]).to(opt.device)

                def get_pred_mod(inp):
                    inputs, lengths, mask, chars, char_lengths = get_pred_emb(inp)
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
                    outputs, _ = C((shared_features, lang_features))
                    return functional.softmax(outputs[:,1,:]).cpu().detach().numpy()

                text = """"""
                for j in range(lengths[i]):
                    text += " " + vocab.get_word(inputs[i][j])
                text = text.lstrip().rstrip()
                for j in range(lengths[i]):
                    if j>0:
                        continue
                    j=1
                    word = vocab.get_word(inputs[i][j])
                    sampler = MaskingTextSampler(replacement="UNK",max_replace=0.7,token_pattern=None,bow=False)
                    samples, similarity = sampler.sample_near(text, n_samples=4)

                    te = TextExplainer(sampler=sampler, position_dependent=True, random_state=12)
                    if tag_vocab.get_tag(targets[i][j]) == tag_vocab.get_tag(pred[i][j]):
                        right_pred = True
                    else:
                        right_pred = False
                    try:
                        te.fit(text, get_pred_mod)
                        htmloutp = eli5.format_as_html(eli5.explain_prediction(te.clf_, te.doc_, vec=te.vec_, target_names=['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']))
                        wandb.log({f'{right_pred}_{name}_explain_nlp_w1': wandb.Html(htmloutp)})
                    except:
                        print(text)
    return None


def main():
    wandb.config.update(opt)
    if not os.path.exists(opt.model_save_file):
        os.makedirs(opt.model_save_file)
    log.info('Running the S-MAN + P-MoE + C-MoE model...')
    vocabs = {}
    tag_vocab = TagVocab()
    assert opt.use_wordemb or opt.use_charemb, "At least one of word or char embeddings must be used!"
    char_vocab = Vocab(opt.charemb_size)
    log.info(f'Loading Datasets...')
    log.info(f'Languages {opt.langs}')

    log.info('Loading Embeddings...')
    train_sets, dev_sets, test_sets, unlabeled_sets = {}, {}, {}, {}
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
                get_dataset_fn(vocabs[lang], char_vocab, tag_vocab, opt.conll_dir, lang)

    opt.num_labels = len(tag_vocab)
    log.info(f'Tagset: {tag_vocab.id2tag}')
    log.info(f'Done Loading Datasets.')

    train(vocabs, char_vocab, tag_vocab, train_sets, dev_sets, test_sets, unlabeled_sets)
    log.info(f'LIME done...')
    return None

if __name__ == '__main__':
    main()
