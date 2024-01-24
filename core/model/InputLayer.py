import logging
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

'''
embedding?
https://wikidocs.net/64779 / 잘 정리되어 있음
임베딩 층의 입력으로 사용하기 위해서 입력 시퀀스의 각 단어들은 모두 정수 인코딩이 되어있어야 함

왜 embedding한 결과에 루트(d_model) 값을 곱해주는가? 에 대한 이유
The reason we increase the embedding values before addition is to make the positional encoding relatively smaller. 
This means the original meaning in the embedding vector won’t be lost when we add them together.
by https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#d554
'''
class Embeddings(nn.Module):
    def __init__(self, d_model=512, vocab=10000):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

'''
논문에서,
Since our model contains no recurrence and no convolution, in order for the model to make use of the
order of the sequence, we must inject some information about the relative or absolute position of the
tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the
bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel
as the embeddings, so that the two can be summed. There are many choices of positional encodings,
learned and fixed [9].

In this work, we use sine and cosine functions of different frequencies:
P E(pos,2i) = sin(pos/10000^(2i/dmodel)) # 짝수
P E(pos,2i+1) = cos(pos/10000^(2i/dmodel)) # 홀수

where pos is the position and i is the dimension. That is, each dimension of the positional encoding
corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We
chose this function because we hypothesized it would allow the model to easily learn to attend by
relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of
P Epos.
We also experimented with using learned positional embeddings [9] instead, and found that the two
versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version
because it may allow the model to extrapolate to sequence lengths longer than the ones encountered
during training.

-정리-
seq2seq 모델같은 경우는 RNN을 사용하므로, 입력 자체에 시간속성이 부여되어 있다. 그런데 transformer같은 경우는 그런게 없다.
그래서 transformer의 encoder, decoder에 시간 속성을 부여하기 위해서 위의 sin, cos 함수를 더해주는 것이다.
'''

class PositionalEncoding(nn.Module):

    '''
    length는 좀 길게 설정한다. 내가 좋아하는 숫자 210
    '''
    def __init__(self, length=5000, d_model=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, length).unsqueeze(1) # (50, 1)

        if d_model % 2 == 1:
            denominator= torch.pow(10000, torch.div(2*torch.arange(d_model//2 + 1), d_model)).unsqueeze(0) # (1, 128)
            sin_denominator = denominator
            cos_denominator = denominator[:,:-1]
        else:
            denominator= torch.pow(10000, torch.div(2*torch.arange(d_model//2), d_model)).unsqueeze(0) # (1, 128)
            sin_denominator =  denominator
            cos_denominator =  denominator

        # if d_model % 2 == 1:
        #     sin_denominator = denominator[:, :d_model//2 + 1]
        #     cos_denominator = denominator[:, :d_model//2]
        # else:
        #     sin_denominator =  denominator[:, :d_model//2]
        #     cos_denominator =  denominator[:, :d_model//2]

        positionalencoding = torch.zeros(length, d_model) # (50, 128)
        positionalencoding[:, 0::2] = torch.sin(torch.div(position, sin_denominator))
        positionalencoding[:, 1::2] = torch.cos(torch.div(position, cos_denominator))
        positionalencoding = positionalencoding.unsqueeze(0) # batch 차원 추가

        '''
        register_buffer 로 layer를 등록하면 어떤 특징이 있는가?
        1. optimizer가 업데이트하지 않는다.
        2. 그러나 값은 존재한다(하나의 layer로써 작용한다고 보면 된다.)
        3. state_dict()로 확인이 가능하다.
        4. GPU연산이 가능하다.
        따라서 네트워크를 구성함에 있어서 네트워크를 end2end로 학습시키고 싶은데 중간에 업데이트를 하지않는 일반 layer를 넣고 싶을 때 사용할 수 있다.
        '''
        self.register_buffer("positionalencoding", positionalencoding)

    def forward(self, x):
        x = self.dropout(x + self.positionalencoding[:, :x.size(1)]) # positionalencoding에 batch 차원 추가 (batch_size=1, 50, 128)
        return x

# padding index를 몇으로 정했냐? 에 따라서 달라질수 있다.
def encoder_mask(x, padding_index = 0):
    '''
    x shape : (batch, sequence_length)
    '''
    assert len(x.size()) == 2 # (batch, sequence_length)와 같은 형태여야 함
    return ((x == padding_index) == False)[:, None, None, :].to(x.dtype) # type_as(x)는 torchscript파일로 추출하지 않는 경우 필요없다.

def decoder_mask(x, padding_index = 0):
    '''
    encoder_mask도 포함한다.
    x shape : (batch, sequence_length)
    '''
    assert len(x.size()) == 2 # (batch, sequence_length)와 같은 형태여야 함
    sequence_length = x.shape[-1]
    e_mask = encoder_mask(x, padding_index=padding_index)
    d_mask = (torch.triu(torch.ones(1, 1, sequence_length, sequence_length), diagonal=1) == 0).type_as(e_mask)
    return e_mask & d_mask

if __name__ == "__main__":

    # PositionalEncoding 그려보기 / 문장의 길이 50, 임베딩 벡터의 차원 128
    pe = PositionalEncoding(210, 128)
    fig = plt.figure(figsize=(6, 6))
    plt.title("Positional Encoding")
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    plt.pcolormesh(pe.positionalencoding[0].numpy(), cmap = plt.cm.PiYG)
    plt.xlabel('Depth')
    plt.xlim((0, 128))
    plt.ylabel('Position')
    plt.colorbar()
    plt.savefig("pe.png")
    plt.show()

    plt.title("decoder mask")
    plt.imshow(decoder_mask(torch.tensor([[3,4,2,3,5,6,11,3,4,1,2,3,6,0]]))[0][0])
    plt.savefig("decoder_mask.png")
    plt.show()
