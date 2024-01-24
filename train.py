import logging
import os
import platform
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import Adam, RMSprop, SGD, lr_scheduler
from tqdm import tqdm

from core import encoder_mask, decoder_mask, Transformer
from core import traindataloader, validdataloader

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

# 초기화 참고하기
# https://pytorch.org/docs/stable/nn.init.html?highlight=nn%20init#torch.nn.init.kaiming_normal_

def run(epoch=100,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        batch_size=256,
        batch_log=100,
        subdivision=1,
        pin_memory=True,
        num_workers=4,
        optimizer="ADAM",
        save_period=5,
        load_period=10,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        weight_decay=0.000001,
        GPU_COUNT=0,
        eval_period=5):

    if GPU_COUNT == 0:
        device = torch.device("cpu")
    elif GPU_COUNT == 1:
        device = torch.device("cuda")
    else:
        device = [torch.device(f"cuda:{i}") for i in range(0, GPU_COUNT)]

    if isinstance(device, (list, tuple)):
        context = device[0]
    else:
        context = device

    # 운영체제 확인
    if platform.system() == "Linux":
        logging.info(f"{platform.system()} OS")
    elif platform.system() == "Windows":
        logging.info(f"{platform.system()} OS")
    else:
        logging.info(f"{platform.system()} OS")

    # free memory는 정확하지 않은 것 같고, torch.cuda.max_memory_allocated() 가 정확히 어떻게 동작하는지?
    if isinstance(device, (list, tuple)):
        for i, d in enumerate(device):
            total_memory = torch.cuda.get_device_properties(d).total_memory
            free_memory = total_memory - torch.cuda.max_memory_allocated(d)
            free_memory = round(free_memory / (1024 ** 3), 2)
            total_memory = round(total_memory / (1024 ** 3), 2)
            logging.info(f'{torch.cuda.get_device_name(d)}')
            logging.info(f'Running on {d} / free memory : {free_memory}GB / total memory {total_memory}GB')
    else:
        if GPU_COUNT == 1:
            total_memory = torch.cuda.get_device_properties(device).total_memory
            free_memory = total_memory - torch.cuda.max_memory_allocated(device)
            free_memory = round(free_memory / (1024 ** 3), 2)
            total_memory = round(total_memory / (1024 ** 3), 2)
            logging.info(f'{torch.cuda.get_device_name(device)}')
            logging.info(f'Running on {device} / free memory : {free_memory}GB / total memory {total_memory}GB')
        else:
            logging.info(f'Running on {device}')

    if GPU_COUNT > 0 and batch_size < GPU_COUNT:
        logging.info("batch size must be greater than gpu number")
        exit(0)

    logging.info("training Transformer")
    train_dataloader, train_dataset = traindataloader(batch_size=batch_size,
                                                      pin_memory=pin_memory,
                                                      num_workers=num_workers)

    train_update_number_per_epoch = len(train_dataloader)
    if train_update_number_per_epoch < 1:
        logging.warning("train batch size가 데이터 수보다 큼")
        exit(0)

    valid_dataloader, valid_dataset = validdataloader(batch_size=batch_size,
                                                      pin_memory=pin_memory,
                                                      num_workers=num_workers)

    valid_update_number_per_epoch = len(valid_dataloader)
    if valid_update_number_per_epoch < 1:
        logging.warning("valid batch size가 데이터 수보다 큼")
        exit(0)

    optimizer = optimizer.upper()

    # https://discuss.pytorch.org/t/how-to-save-the-optimizer-setting-in-a-log-in-pytorch/17187
    weight_path = os.path.join("weights", "transformer")
    param_path = os.path.join(weight_path, f'{load_period:04d}.pt')

    start_epoch = 0
    net = Transformer(src_vocab=train_dataset.SRC_VOCAB_SIZE, tgt_vocab=train_dataset.TGT_VOCAB_SIZE,
                      d_model=d_model,
                      nhead=nhead,
                      num_encoder_layers=num_encoder_layers,
                      num_decoder_layers=num_decoder_layers,
                      dim_feedforward=dim_feedforward,
                      dropout=dropout)

    # weight 초기화 - 이렇게 초기화 하는 것이 편하다.
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if os.path.exists(param_path):
        start_epoch = load_period
        checkpoint = torch.load(param_path)
        if 'model_state_dict' in checkpoint:
            try:
                net.load_state_dict(checkpoint['model_state_dict'])
            except Exception as E:
                logging.info(E)
            else:
                logging.info(f"loading model_state_dict")

    if start_epoch + 1 >= epoch + 1:
        logging.info("this model has already been optimized")
        exit(0)

    net.to(context)

    if optimizer.upper() == "ADAM":
        trainer = Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif optimizer.upper() == "RMSPROP":
        trainer = RMSprop(net.parameters(), lr=learning_rate, alpha=0.99, weight_decay=weight_decay, momentum=0)
    elif optimizer.upper() == "SGD":
        trainer = SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        logging.error("optimizer not selected")
        exit(0)

    if os.path.exists(param_path):
        # optimizer weight 불러오기
        checkpoint = torch.load(param_path)
        if 'optimizer_state_dict' in checkpoint:
            try:
                trainer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as E:
                logging.info(E)
            else:
                logging.info(f"loading optimizer_state_dict")

    if isinstance(device, (list, tuple)):
        net = DataParallel(net, device_ids=device, output_device=context, dim=0)

    # padding index는 제외 시킨다.
    CELoss = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.PAD_IDX, reduction="mean")

    # optimizer
    # https://pytorch.org/docs/master/optim.html?highlight=lr%20sche#torch.optim.lr_scheduler.CosineAnnealingLR
    unit = 1 if train_update_number_per_epoch < 1 else train_update_number_per_epoch
    step = unit * decay_step
    lr_sch = lr_scheduler.StepLR(trainer, step, gamma=decay_lr, last_epoch=-1)

    # torch split이 numpy, mxnet split과 달라서 아래와 같은 작업을 하는 것
    if batch_size % subdivision == 0:
        chunk = int(batch_size) // int(subdivision)
    else:
        logging.info(f"batch_size / subdivision 이 나누어 떨어지지 않습니다.")
        logging.info(f"subdivision 을 다시 설정하고 학습 진행하세요.")
        exit(0)

    start_time = time.time()
    for i in tqdm(range(start_epoch + 1, epoch + 1, 1), initial=start_epoch + 1, total=epoch):

        net.train()
        loss_sum = 0
        time_stamp = time.time()

        '''
            Multi30k 데이터셋에 __iter__로 초기화 구현이 안되있어서, 한 epoch 끝날때마다" 
            dataloader 다시 만들어줘야한다..
        '''
        train_dataloader, train_dataset = traindataloader(batch_size=batch_size,
                                                          pin_memory=pin_memory,
                                                          num_workers=num_workers)

        # multiscale을 하게되면 여기서 train_dataloader을 다시 만드는 것이 좋겠군..
        for batch_count, (src, tgt) in enumerate(
                train_dataloader,
                start=1):

            trainer.zero_grad()

            src = src.to(context)
            '''
            이렇게 하는 이유?
            209 line에서 net = net.to(context)로 함
            gpu>=1 인 경우 net = DataParallel(net, device_ids=device, output_device=context, dim=0) 에서 
            output_device - gradient가 계산되는 곳을 context로 했기 때문에 아래의 target들도 context로 지정해줘야 함
            '''
            tgt = tgt.to(context)

            src_split = torch.split(src, chunk, dim=0)
            tgt_split = torch.split(tgt, chunk, dim=0)

            losses = []

            for src_part_input, tgt_part in zip(
                    src_split,
                    tgt_split):

                tgt_part_input = tgt_part[:, :-1]
                tgt_part_out = tgt_part[:, 1:]

                src_mask = encoder_mask(src_part_input, padding_index = train_dataset.PAD_IDX)
                tgt_mask = decoder_mask(tgt_part_input, padding_index = train_dataset.PAD_IDX)
                pred = net(src_part_input, tgt_part_input, src_mask, tgt_mask)
                '''
                pytorch는 trainer.step()에서 batch_size 인자가 없다.
                Loss 구현시 고려해야 한다.(mean 모드) 
                '''

                # sequence를 batch 쪽으로 합쳐서 계산.
                loss = torch.div(CELoss(pred.reshape(-1, train_dataset.TGT_VOCAB_SIZE), tgt_part_out.reshape(-1)), subdivision)
                loss.backward()
                losses.append(loss.item())

            trainer.step()
            lr_sch.step()
            loss_sum += sum(losses)

            if batch_count % batch_log == 0:
                logging.info(f'[Epoch {i}][Batch {batch_count}/{train_update_number_per_epoch}]'
                             f'[Speed {src.shape[0] / (time.time() - time_stamp):.3f} samples/sec]'
                             f'[Lr = {lr_sch.get_last_lr()}]'
                             f'[loss = {sum(losses):.3f}]')
            time_stamp = time.time()

        train_loss_mean = np.divide(loss_sum, train_update_number_per_epoch)

        logging.info(
            f"train loss : {train_loss_mean}")

        if i % save_period == 0:

            if not os.path.exists(weight_path):
                os.makedirs(weight_path)

            net = net.module if isinstance(device, (list, tuple)) else net

            try:
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': trainer.state_dict()}, os.path.join(weight_path, f'{i:04d}.pt'))

                script = torch.jit.script(net)
                script.save(os.path.join(weight_path, f'{i:04d}.jit'))

            except Exception as E:
                logging.error(f"pt, jit export 예외 발생 : {E}")
            else:
                logging.info("pt, jit export 성공")

        if i % eval_period == 0:

            net.eval()
            loss_sum = 0

            valid_dataloader, valid_dataset = validdataloader(batch_size=batch_size,
                                                              pin_memory=pin_memory,
                                                              num_workers=num_workers)

            for src, tgt in valid_dataloader:
                src = src.to(context)
                tgt = tgt.to(context)

                tgt_input = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                src_mask = encoder_mask(src, padding_index = valid_dataset.PAD_IDX)
                tgt_mask = decoder_mask(tgt_input, padding_index = train_dataset.PAD_IDX)
                with torch.no_grad():
                    pred = net(src, tgt_input, src_mask, tgt_mask)

                loss = CELoss(pred.reshape(-1, valid_dataset.TGT_VOCAB_SIZE), tgt_out.reshape(-1))
                loss_sum += loss.item()

            valid_loss_mean = np.divide(loss_sum, valid_update_number_per_epoch)

            logging.info(
                f"valid loss : {valid_loss_mean}")

    end_time = time.time()
    learning_time = end_time - start_time
    logging.info(f"learning time : 약, {learning_time / 3600:0.2f}H")
    logging.info("optimization completed")

if __name__ == "__main__":
    run(epoch=100,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        batch_size=16,
        batch_log=100,
        subdivision=1,
        pin_memory=True,
        num_workers=4,
        optimizer="ADAM",
        save_period=5,
        load_period=10,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        weight_decay=0.000001,
        GPU_COUNT=0,
        eval_period=5)
