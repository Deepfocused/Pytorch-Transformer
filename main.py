import logging

import torch
import yaml

import train

# nms 구현하면 끝
stream = yaml.load(open("configs/transformer.yaml", "rt", encoding='UTF8'), Loader=yaml.SafeLoader)

# model
parser = stream['model']

d_model = parser['d_model']
nhead = parser['nhead']
num_encoder_layers = parser['num_encoder_layers']
num_decoder_layers = parser['num_decoder_layers']
dim_feedforward = parser['dim_feedforward']
dropout = parser['dropout']

training = parser["training"]
save_period = parser["save_period"]
load_period = parser["load_period"]

# hyperparameters
parser = stream['hyperparameters']
epoch = parser["epoch"]
batch_size = parser["batch_size"]
batch_log = parser["batch_log"]
subdivision = parser["subdivision"]
pin_memory = parser["pin_memory"]
num_workers = parser["num_workers"]
optimizer = parser["optimizer"]
learning_rate = parser["learning_rate"]
weight_decay = parser["weight_decay"]
decay_lr = parser["decay_lr"]
decay_step = parser["decay_step"]

parser = stream['validation']
eval_period = parser["eval_period"]

# gpu vs cpu
parser = stream['context']
using_cuda = parser["using_cuda"]

if torch.cuda.device_count() > 0 and using_cuda:
    GPU_COUNT = torch.cuda.device_count()
else:
    GPU_COUNT = 0

# window 운영체제에서 freeze support 안나오게 하려면, 아래와 같이 __name__ == "__main__" 에 해줘야함.
if __name__ == "__main__":

    print("\n실행 경로 : " + __file__)
    if training:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True # 그래프가 변하는 경우 학습 속도 느려질수 있음.
        train.run(epoch=epoch,
                  d_model=d_model,
                  nhead=nhead,
                  num_encoder_layers=num_encoder_layers,
                  num_decoder_layers=num_decoder_layers,
                  dim_feedforward=dim_feedforward,
                  dropout=dropout,
                  batch_size=batch_size,
                  batch_log=batch_log,
                  subdivision=subdivision,
                  pin_memory=pin_memory,
                  num_workers=num_workers,
                  optimizer=optimizer,
                  save_period=save_period,
                  load_period=load_period,
                  learning_rate=learning_rate, decay_lr=decay_lr, decay_step=decay_step,
                  weight_decay=weight_decay,
                  GPU_COUNT=GPU_COUNT,
                  eval_period=eval_period)
    else:
        logging.info(f"training = {training} 입니다.")
