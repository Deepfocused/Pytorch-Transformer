'''
https://pytorch.org/tutorials/beginner/translation_transformer.html 참고
Multi30k 데이터셋 사용
'''

from typing import Iterable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator


# Dataset
class Dataset(object):
    def __init__(self, ):

        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

        self.token_transform = {}
        self.vocab_transform = {}
        self.sequential_transform = {}

        # 다른 데이터셋을 사용할 경우 아래 토크나이저 바꾸면 됨
        self.SRC_LANGUAGE = 'de'
        self.TGT_LANGUAGE = 'en'
        self.token_transform[self.SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
        self.token_transform[self.TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

        # 다른 데이터셋을 사용할 경우 아래 Multi30K부분만 처리하면 됨
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            # Training data Iterator
            train_iter = Multi30k(split='train', language_pair=(self.SRC_LANGUAGE, self.TGT_LANGUAGE))
            # Create torchtext's Vocab object
            self.vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(train_iter, ln),
                                                                 min_freq=1,
                                                                 # The minimum frequency needed to include a token in the vocabulary.
                                                                 specials=self.special_symbols,
                                                                 special_first=True)  # Indicates whether to insert symbols at the beginning or at the end.

        # Set UNK_IDX as the default index. This index is returned when the token is not found.
        # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            self.vocab_transform[ln].set_default_index(self.UNK_IDX)

        self.SRC_VOCAB_SIZE = len(self.vocab_transform[self.SRC_LANGUAGE])
        self.TGT_VOCAB_SIZE = len(self.vocab_transform[self.TGT_LANGUAGE])

        # src and tgt language text transforms to convert raw strings into tensors indices
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:

            self.sequential_transform[ln] = self.sequential_transforms(self.token_transform[ln],  # Tokenization
                                                                       self.vocab_transform[ln],  # Numericalization
                                                                       self.tensor_transform)  # Add BOS/EOS and create tensor


        # 이터레이터라서...한번돌면 끝난다..
        self.train_dataset = Multi30k(split='train', language_pair=(self.SRC_LANGUAGE, self.TGT_LANGUAGE))
        self.valid_dataset = Multi30k(split='valid', language_pair=(self.SRC_LANGUAGE, self.TGT_LANGUAGE))
        self.test_dataset = Multi30k(split='test', language_pair=(self.SRC_LANGUAGE, self.TGT_LANGUAGE))

    # helper function to yield list of tokens
    def yield_tokens(self, data_iter: Iterable, language: str) -> List[str]:
        language_index = {self.SRC_LANGUAGE: 0, self.TGT_LANGUAGE: 1}
        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[language_index[language]])

    # helper function to club together sequential operations
    # 순차 적용
    #Tokenization, Numericalization, Add BOS/EOS and create tensor 순자 적용
    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([self.EOS_IDX])))

    # 위의 모든 것을 collate_fn을 위한 것들
    # function to collate data samples into batch tensors
    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.sequential_transform[self.SRC_LANGUAGE](src_sample.rstrip("\n"))) #줄 바꿈 없애기
            tgt_batch.append(self.sequential_transform[self.TGT_LANGUAGE](tgt_sample.rstrip("\n")))

        ''' 
            https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
            Pad a list of variable length Tensors with padding_value
            pad_sequence stacks a list of Tensors along a new dimension, and pads them to equal length. For example,
            if the input is list of sequences with size L x * and if batch_first is False, and T x B x * otherwise.
            B is batch size. It is equal to the number of elements in sequences. 
            T is length of the longest sequence. L is length of the sequence. * is any number of trailing dimensions, including none.
            >>> from torch.nn.utils.rnn import pad_sequence
            >>> a = torch.ones(25, 300)
            >>> b = torch.ones(22, 300)
            >>> c = torch.ones(15, 300)
            >>> pad_sequence([a, b, c]).size()
            torch.Size([25, 3, 300])
        '''

        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=self.PAD_IDX)
        return src_batch, tgt_batch

if __name__ == "__main__":

    dataset = Dataset()
    print(next(dataset.train_dataset))

