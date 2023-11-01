import math

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
MultiHeadedAttention의 장점?
코드상으로는 weight를 n개로 쪼개서 attention 값을 병렬로 계산하는 것

논문상으로는
Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions. With a single attention head, averaging inhibits this.
<번역>
Multi-head attention은 모델이 서로 다른 위치의 서로 다른 표현 하위 공간으로부터의 정보에 공동으로 attention이 가능하다. 
단일 attention으로 평균을 계산하는 것을 그러한 동작을 억제한다.
'''
# attention layer
class MultiHeadedAttention(nn.Module):
    def __init__(self, nhead=8, d_model=512):

        super(MultiHeadedAttention, self).__init__()
        assert d_model % nhead == 0

        self.nhead = nhead
        self.d_k = d_model // nhead

        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.q_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

        self.attn_weight = None # 보여주기 위함

    # Scaled Dot-Product Attention
    def attention(self, value, key, query, mask):

        '''
        어텐션 스코어(Attention Score)에 Softmax를 적용 어텐션 분포(Attention Distribution or Attetion Weight)를 구하고,
        각 V 벡터와 가중합하여 어텐션 값(Attention Value)을 구하는 과정

        '''
        d_k = value.size(-1)

        # query shape : (batch size, nhead, sequence length, d_model // nhead)
        # transposed key shape : (batch size, nhead, d_model // nhead, sequence length)
        # result shape : (batch size, nhead, sequence length, sequence length)
        attr_score= torch.div(torch.matmul(query, key.transpose(-2, -1)), math.sqrt(d_k))

        if mask is not None:
            attr_score = attr_score.masked_fill_(mask == 0, -1e9)

        # softmax 는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
        # attention distribution or attention weight(논문)
        # attr_weight : (batch_size, nhead, sequence length(query 길이), sequence length(key 길이))
        attr_weight = F.softmax(attr_score, dim=-1)

        # Attention 값
        # attr_value shape : (batch size, nhead, sequence length(query 길이), d_model // nhead)
        attr_value = torch.matmul(attr_weight, value)
        return attr_value, attr_weight


    def forward(self, value, key, query, mask):
        '''
        query shape = (batch_size, vocab_size, d_model)
        key shape = (batch_size, vocab_size, d_model)
        value shape = = (batch_size, vocab_size, d_model)
        '''

        batch_size = value.size(0)

        # 1. embedding vector를 N개의 Head로 자른다.
        # 아래의 3줄 shape : (batch_size, sequence length, d_model) -> (batch_size, sequence length, nhead, d_model // nhead)
        value = self.v_linear(value).view(batch_size, -1, self.nhead, self.d_k)
        key = self.k_linear(key).view(batch_size, -1, self.nhead, self.d_k)
        query = self.q_linear(query).view(batch_size, -1, self.nhead, self.d_k)

        # 2. 1번축과 2번축의 위치를 바꾼다.
        # 아래의 3줄 shape : (batch size, nhead, sequence length, d_model // nhead)
        value = value.transpose(1, 2)
        key = key.transpose(1,2)
        query = query.transpose(1, 2)

        # 2) attention 적용
        # attr_value shape : (batch size, nhead, sequence length(query 길이), d_model // nhead)
        attr_value, self.attn_weight = self.attention(key, value, query, mask=mask)

        # 3) concat하고 out layer 통과시키기
        # attr_value shape : (batch size, sequence length(query 길이), d_model)
        '''
            contiguous 필수!!!
            view(), expand(), transpose(), permute(), narrow(), etc
            와 같은 함수는 메모리 공유함. 축 바꿀시 문제생길 수 있음
            
            참고 : https://f-future.tistory.com/entry/Pytorch-Contiguous
        '''
        attr_value = attr_value.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)

        # attr_value = attr_value.transpose(1, 2).view(batch_size, -1, self.h * self.d_k)
        return self.out(attr_value) # (batch size, sequence length(query 길이), d_model)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model=512, dim_feedforward=2048):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, dim_feedforward)
        self.w_2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))
