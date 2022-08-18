import os

import torch
from torch.utils.data import Dataset

from options import opt

class EyeTracking(Dataset):
    """
        Idea is to fill X with the words and Y1..Y4 with the gaze features
    """
    def __init__(self, input_file, vocab, char_vocab, update_vocab):
        self.raw_X = []
        self.raw_Y = []
        with open(input_file) as inf:
            next(inf)
            for line in inf:
                line = line.split('\t')
                self.raw_X.append(str(line[1]))
                self.raw_Y.append([float(line[x]) for x in range(3,3 + opt.n_gaze_feat)])
                if vocab and update_vocab:
                    vocab.add_word(line[1])
                if char_vocab and update_vocab:
                    for ch in line[1]:
                        char_vocab.add_word(ch, normalize=opt.lowercase_char)
        self.X = []
        for xs in self.raw_X:
            x = {}
            if vocab:
                x['words'] = [vocab.lookup(w) for w in xs]
            if char_vocab:
                x['chars'] = [[char_vocab.lookup(ch, normalize=opt.lowercase_char) for ch in w] for w in xs]
            self.X.append(x)
        assert len(self.X) == len(self.raw_Y)

    def __len__(self):
        return len(self.raw_Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.raw_Y[idx])

def get_gaze_data(vocab, char_vocab, data_dir, corpus, lang):
    print(f'Loading {corpus} Corpus for {lang} Language..')
    train_set = EyeTracking(os.path.join(data_dir, f'{lang}.{corpus}'),
                            vocab, char_vocab, update_vocab=True)
    return train_set

