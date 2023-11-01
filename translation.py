import logging
import os

import torch

from core import encoder_mask, decoder_mask
from core import traindataloader

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

def greedy_decode(net, src, src_mask, START_INDEX=2, END_INDEX=3, PADDING_INDEX=1):

    # encoder 추론
    embedded_src = net.encoder_PE(net.encoder_embedded(src))
    from_encoder = net.Encoders(embedded_src, src_mask)

    # Inference의 경우 decoder에는 일단 START_INDEX로 구성된 sequence 한개의 배열만 넣는다.
    tgt = torch.ones(1, 1).fill_(START_INDEX).type_as(src)
    for i in range(src.shape[-1]-1):
        tgt_mask = decoder_mask(tgt, padding_index=PADDING_INDEX)

        # decoder 추론
        embedded_tgt = net.decoder_PE(net.decoder_embedded(tgt))
        output = net.Decoders(embedded_tgt, from_encoder, src_mask, tgt_mask)

        # 행에서 가장 아래의 것(결과 or 현재)만 가져온다.
        prediction = torch.softmax(net.output(output), dim=-1)[:, -1]
        _, next_word = torch.max(prediction, dim=-1) # max, max_indices 반환
        next_word = next_word.item()

        # 행 방향으로 결과를 쌓는다. -> 다시 decoder의 입력이 된다.
        tgt = torch.cat([tgt, torch.ones(1, 1).fill_(next_word).type_as(src)], dim=-1)

        # end index를 만나면 for문을 벗어난다.
        if next_word == END_INDEX:
            break

    return tgt

def translate(net: torch.nn.Module, src_sentence: str, device: object):

    _, dataset = traindataloader()
    src = dataset.sequential_transform[dataset.SRC_LANGUAGE](src_sentence).view(1, -1).to(device) # (batch, sequence length)
    src_mask = encoder_mask(src, padding_index=dataset.PAD_IDX) # 마스크 생성
    tgt_tokens = greedy_decode(
        net, src, src_mask,
        START_INDEX=dataset.BOS_IDX,
        END_INDEX=dataset.EOS_IDX,
        PADDING_INDEX=dataset.PAD_IDX).flatten()

    start_token = dataset.special_symbols[-2] # '<bos>'
    enc_token = dataset.special_symbols[-1] # '<eos>'
    # torchtext 매우 유용하다
    return " ".join(dataset.vocab_transform[dataset.TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace(start_token, "").replace(enc_token, "")

if __name__ == "__main__":

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    jit_number = 105

    if torch.cuda.device_count() > 0 :
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logging.info("jit model test")
    weight_path = os.path.join("weights", "transformer")
    try:
        net = torch.jit.load(os.path.join(weight_path, f'{jit_number:04d}.jit'), map_location=device)
        net.eval()
    except Exception:
        # DEBUG, INFO, WARNING, ERROR, CRITICAL 의 5가지 등급
        logging.info("loading jit 실패")
        exit(0)
    else:
        logging.info("loading jit 성공")

    '''
        트랜스포머에서는 Key의 경우에 <PAD> 토큰이 존재한다면 이에 대해서는 유사도를 구하지 않도록 마스킹(Masking)을 해줌
        여기서 마스킹이란 어텐션에서 제외하기 위해 값을 가린다는 의미
        어텐션 스코어 행렬에서 행에 해당하는 문장은 Query, 열에 해당하는 문장은 Key
        그리고 Key에 <PAD>가 있는 경우에는 해당 열 전체를 마스킹을 해줌
        마스킹을 하는 방법은 어텐션 스코어 행렬의 마스킹 위치에 매우 작은 음수값을 넣어주면 됨.
        from https://wikidocs.net/31379
    '''

    # 번역 시작
    with torch.no_grad():
        #translate(net, "Eine Gruppe von Menschen steht vor einem Iglu .", device)
        print(translate(net, "Eine Gruppe von Menschen steht vor einem Iglu .", device))
        # result : A group of people standing in front of an igloo
        # 구글 번역기 : A group of people stands in front of an igloo
        print(translate(net, "Vor dem Gebäude stehen mehrere Personen.", device))
        # result : There is several people standing in front of
        # 구글 번역기 : Several people are standing in front of the building