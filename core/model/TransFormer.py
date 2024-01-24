import logging
import os

import torch
import torch.nn as nn

from core.model.InputLayer import PositionalEncoding, Embeddings, encoder_mask, decoder_mask
from core.model.Sublayers import MultiHeadedAttention, PositionwiseFeedForward

'''
reference1 https://nlp.seas.harvard.edu/2018/04/03/attention.html 전반적인 내용이 잘 담겨있으나, 이해가 어려운 코드들도 많습니다.
reference2 https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#d554 위의 코드를 기반으로 자세히 설명해준 편입니다.
reference3 https://wikidocs.net/31379 : 도움이 많이 됬습니다.
reference4 https://pytorch.org/tutorials/beginner/translation_transformer.html : 공식 튜토리얼? : 데이터 처리에 도움을 많이 받음
reference5 https://sincerechloe.tistory.com/70 : Label smoothing관련 내용 : 참고하여 구현하진 않고, 읽어만 봄.
'''

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

class EncoderLayer(nn.Module):

    # 그림과 똑같이 구성함
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):

        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(nhead=nhead, d_model=d_model)
        self.layernorm1 = nn.LayerNorm(d_model) # shape = (d_model,)
        self.dropout1 = nn.Dropout(p=dropout)

        self.feed_forward = PositionwiseFeedForward(d_model=d_model, dim_feedforward=dim_feedforward)
        self.layernorm2 = nn.LayerNorm(d_model) # shape = (d_model,)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, mask):

        # x shape : (1, vocab_size, d_model)
        # 공식구현에서 Layernorm 적용하는 부분이 바뀌었다고 하는데, 나는 논문대로 하겠다
        x = self.layernorm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
        x = self.layernorm2(x + self.dropout2(self.feed_forward(x)))
        return x

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(nhead=nhead, d_model=d_model)
        self.layernorm1 = nn.LayerNorm(d_model) # shape = (512,)
        self.dropout1 = nn.Dropout(p=dropout)

        self.src_attn = MultiHeadedAttention(nhead=nhead, d_model=d_model)
        self.layernorm2 = nn.LayerNorm(d_model) # shape = (512,)
        self.dropout2 = nn.Dropout(p=dropout)

        self.feed_forward = PositionwiseFeedForward(d_model=d_model, dim_feedforward=dim_feedforward)
        self.layernorm3 = nn.LayerNorm(d_model) # shape = (512,)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, from_encoder, src_mask, tgt_mask):

        # 공식구현에서 Layernorm 적용하는 부분이 바뀌었다고 하는데, 나는 논문대로 하겠다
        x = self.layernorm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask))) # value, key, query 순서
        x = self.layernorm2(x + self.dropout2(self.src_attn(from_encoder, from_encoder, x, src_mask))) # value, key, query 순서
        x = self.layernorm3(x + self.dropout3(self.feed_forward(x)))
        return x


class Encoders(nn.Module):

    def __init__(self, layers):
        super(Encoders, self).__init__()
        self.layers = layers

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoders(nn.Module):

    def __init__(self, layers):
        super(Decoders, self).__init__()
        self.layers = layers

    def forward(self, x, from_encoder, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, from_encoder, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):

    def __init__(self, src_vocab=50, tgt_vocab=50, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder_embedded = Embeddings(d_model=d_model, vocab=src_vocab)
        self.decoder_embedded = Embeddings(d_model=d_model, vocab=tgt_vocab)
        self.encoder_PE = PositionalEncoding(length=src_vocab, d_model=d_model, dropout=dropout)
        self.decoder_PE = PositionalEncoding(length=tgt_vocab, d_model=d_model, dropout=dropout)
        self.Encoders = Encoders(nn.ModuleList([EncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout) for _ in range(num_encoder_layers)]))
        self.Decoders = Decoders(nn.ModuleList([DecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout) for _ in range(num_decoder_layers)]))


        '''
        nn.Linear : 가중치 저정하는 형태가 반대였음.!!! - 주의하기
        Applies a linear transformation to the incoming data: y = xA^T + b 
        This module supports TensorFloat32.
        '''
        self.output = nn.Linear(d_model, tgt_vocab) # weight shape : (tgt_vocab, d_model)

        # 아래와 같이 초기화 하는경우, 위의 각 클래스에가서 아래와 같은 방법으로 직접 다 초기화 해줘야함.
        # for m in self.modules():
        #     if isinstance(m, Conv2d):
        #         torch.nn.init.normal_(m.weight, mean=0., std=0.01)
        #         if m.bias is not None:
        #             torch.nn.init.constant_(m.bias, 0)

    def forward(self, src, tgt, src_mask, tgt_mask):

        x = self.encoder_PE(self.encoder_embedded(src)) # encoder 입력
        y = self.decoder_PE(self.decoder_embedded(tgt)) # decoder 입력
        x = self.Encoders(x, src_mask)
        y = self.Decoders(y, x, src_mask, tgt_mask)
        #return F.softmax(self.output(y), dim=-1)
        return self.output(y) # torch.nn.CrossEntropyLoss 을 사용하므로, raw output을 내보낸다.

if __name__ == "__main__":

    # 브로드 캐스팅을 잘 사용해야한다.
    src_vocab = 5000
    tgt_vocab = 4999
    src_sequence_size = 30
    target_sequence_size = 29
    device = torch.device("cpu")
    net = Transformer(src_vocab=src_vocab, tgt_vocab=tgt_vocab, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                      dim_feedforward=2048, dropout=0.1)
    # weight 초기화 - 이렇게 초기화 하는 것이 편하다.
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    net.eval()
    net.to(device)
    '''
    mask에 대한 설명 
    src_mask, tgt_mask 두 종류가 있는데, 
    https://wikidocs.net/31379 의 패딩마스크 ,룩-어헤드 마스크
    https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#d554 
    의 Creating Our Masks 를 읽어보면 된다.
    '''
    src = torch.randint(0, src_sequence_size, size=(3, src_sequence_size), device=device)
    tgt = torch.randint(0, target_sequence_size, size=(3, target_sequence_size), device=device)

    '''
        트랜스포머에서는 Key의 경우에 <PAD> 토큰이 존재한다면 이에 대해서는 유사도를 구하지 않도록 마스킹(Masking)을 해줌
        여기서 마스킹이란 어텐션에서 제외하기 위해 값을 가린다는 의미
        어텐션 스코어 행렬에서 행에 해당하는 문장은 Query, 열에 해당하는 문장은 Key
        그리고 Key에 <PAD>가 있는 경우에는 해당 열 전체를 마스킹을 해줌
        마스킹을 하는 방법은 어텐션 스코어 행렬의 마스킹 위치에 매우 작은 음수값을 넣어주면 됨.
        from https://wikidocs.net/31379
    '''
    src_mask = encoder_mask(src)

    # 현재 시점의 예측에서 현재 시점보다 미래에 있는 단어들을 참고하지 못하도록
    # src_mask가 포함 됨.
    tgt_mask = decoder_mask(tgt)

    with torch.no_grad():
        output = net(src, tgt, src_mask, tgt_mask)

    # script = torch.jit.script(net)
    # script.save("transformer.jit")
    #
    # net =torch.jit.load("transformer.jit", map_location=device)
    with torch.no_grad():
        output = net(src, tgt, src_mask, tgt_mask)

    print(f"< output shape : {output.shape} >")
    '''
    < output shape : torch.Size([3, 29, 4999]) >
    '''
