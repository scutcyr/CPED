# coding=utf-8
# Copyright 2021 South China University of Technology and 
# Engineering Research Ceter of Ministry of Education on Human Body Perception.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Model training code
# File: train_model.py
# Used for training model
# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2022.04.06

# 关键包版本说明：
# pytorch: 1.9.0+
# pytorch-ignite: 0.4.8
# transformers: 4.18.0

import os
import json
import time
import math
import torch
import socket
import random
import logging
import numpy as np
from pprint import pformat
from argparse import ArgumentParser # 用于函数文件传递参数
from torch.optim.lr_scheduler import LambdaLR, CyclicLR, OneCycleLR
from torch.nn.parallel import DistributedDataParallel # 用于分布式模型训练
from torch.cuda.amp import autocast as autocast # 用于使用自动混合精度，要求torch版本为1.6+
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping, Checkpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LRScheduler
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler

from transformers import (WEIGHTS_NAME, CONFIG_NAME, BertTokenizer, OpenAIGPTTokenizer, OpenAIGPTConfig, GPT2Config) # , AdamW

from torch.optim import AdamW

# 本项目自主撰写的python包
from models import (count_trainable_parameters, count_total_parameters, show_trainable_parameters, freeze_by_model_name, unfreeze_by_model_name)
# 从transformers的代码修改，适配混合精度训练
from models.gpt import OpenAIGPTLMHeadModel
from models.gpt2 import GPT2LMHeadModel
# 读取数据集的dataloader构建函数
from utils import build_cped_dataloaders

logger = logging.getLogger(__file__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2022)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def score_function(engine):
    '''最小化ppl，也就是最大化负的ppl
    
    '''
    return -engine.state.metrics["average_ppl"]



def train():
    '''train()
    封装好的训练模型过程

    '''
    # 参数定义
    parser = ArgumentParser()
    # 模型类型以及路径
    parser.add_argument("--model_type", type=str, default="GPT", choices=['GPT', 'GPT-2'], help="Type of Model(模型类型名称)")
    parser.add_argument("--model_checkpoint", type=str, default="/home/phd-chen.yirong/PretrainedModel/models.huggingface.co/bert-base-chinese", help="Path or URL of the model")
    parser.add_argument("--gpt_model_checkpoint", type=str, default="/home/phd-chen.yirong/PretrainedModel/CDial-GPT/LCCD_GPT_FOR_GPTSPEAKERROBOT", help="Path or URL of the GPT model used to initialized the GPT part of the UiBot.")
    parser.add_argument("--bert_model_checkpoint", type=str, default="./runs/SPEAKERBERT", help="Path or URL of the BERT model used to initialized the BERT part of the UiBot.")  
    parser.add_argument('--log_file', '-log_file', type=str, default="./logs", help="Output logs to a file under this path")

    # 数据集名称、路径等配置
    parser.add_argument("--dataset_name", type=str, default="CPED", choices=['CPED', 'MELD', 'CPED-shuffle'], help="Name of Dataset(数据集名称)")
    parser.add_argument("--data_path", type=str, default="./data/CPED/", help="dir of the dataset(数据集保存路径，可以是目录或者文件名)")
    parser.add_argument("--cache_path", type=str, default="./data/CPED_cache_for_CpedDataset", help="path of the dataset cache(数据集缓存文件的保存路径，必须为文件名)")
    parser.add_argument("--use_speaker_name_as_speaker_list", action='store_true',
                        help="If true using speaker name as speaker_list")
    parser.add_argument("--emotion_type", type=str, default="Emotion", choices=['Sentiment', 'BaseEmotion', 'Emotion'], help="Type of Emotion")
    parser.add_argument("--da_type", type=str, default="DA", choices=['DA', 'BaseDA'], help="Type of DA")
    parser.add_argument('--with_emotion', action='store_true', help="use emotion as token_type")
    parser.add_argument('--with_da', action='store_true', help="use da as token_type")
    parser.add_argument('--with_current_speaker', action='store_true', help="use current speaker as control signal")
    parser.add_argument('--with_current_persona', action='store_true', help="use current persona as control signal")
    parser.add_argument('--with_current_emotion', action='store_true', help="use current emotion as control signal")
    parser.add_argument('--with_current_da', action='store_true', help="use current da as control signal")
    parser.add_argument('--set_eda_in_speaker', action='store_true', help="set eda in speaker")
    parser.add_argument('--set_current_speaker_mask', action='store_true', help="set current_speaker_mask")

    # 训练模型配置
    parser.add_argument('--find_unused_parameters', action='store_true', help="If True find_unused_parameters")
    parser.add_argument('--show_parameters', action='store_true', help="If True show model parameters")
    parser.add_argument("--from_step", type=int, default=-1, help="Init learning rate from this step")
    parser.add_argument('--pretrained', action='store_true', help="If False train from scratch")
    ## Adamw优化器参数配置：正则化、beta1、beta2、eps数值设置
    parser.add_argument('--L2_regularization', action='store_true', help="If False train without L2 Regularization")
    parser.add_argument("--L2_weight_decay", type=float, default=1e-2, help="L2 weight decay")
    parser.add_argument("--adamw_beta1", type=float, default=0.9, help="Adam's beta1 parameter")
    parser.add_argument("--adamw_beta2", type=float, default=0.999, help="Adam's beta2 parameter")
    parser.add_argument("--adamw_eps", type=float, default=1e-6, help="Adam's epsilon for numerical stability")

    # 损失函数的各部分占比
    parser.add_argument("--alpha_nll", type=float, default=1.0, help="alpha_nll")
    parser.add_argument("--alpha_emotion", type=float, default=1.0, help="alpha_emotion")
    parser.add_argument("--alpha_da", type=float, default=1.0, help="alpha_da")
    parser.add_argument("--alpha_per_gen", type=float, default=1.0, help="alpha_per_gen")
    parser.add_argument("--alpha_per_neu", type=float, default=1.0, help="alpha_per_neu")
    parser.add_argument("--alpha_per_ext", type=float, default=1.0, help="alpha_per_ext")
    parser.add_argument("--alpha_per_ope", type=float, default=1.0, help="alpha_per_ope")
    parser.add_argument("--alpha_per_agr", type=float, default=1.0, help="alpha_per_agr")
    parser.add_argument("--alpha_per_con", type=float, default=1.0, help="alpha_per_con")

    # 冻结模型的部分层
    parser.add_argument('--freeze_model', action='store_true', help="If True freeze some layers of the model")
    parser.add_argument('--freeze_start_layer', type=int, default=0, help="冻结指定的层范围，格式为start-end，其中start取值范围为0~11，end取值范围为0~11，start<=end")
    parser.add_argument('--freeze_end_layer', type=int, default=11, help="冻结指定的层范围，格式为start-end，其中start取值范围为0~11，end取值范围为0~11，start<=end")


    parser.add_argument("--not_return_dict", action='store_true', help="If False the model return dict as result（适配最新版本transformers的模型输出）")
    parser.add_argument("--do_unscale", action='store_true', help="Calling scaler.unscale_(optimizer) before clipping enables you to clip unscaled gradients as usual（梯度裁剪）")
    parser.add_argument("--retain_graph", action='store_true', help="If true using retain_graph=True in loss.backward")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of subprocesses for data loading")
    parser.add_argument("--n_epochs", type=int, default=32, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--max_history", type=int, default=25, help="Number of previous exchanges to keep in history")
    
    parser.add_argument("--n_emd", type=int, default=768, help="Number of n_emd in config file (for noam)")
    # 学习率设置
    parser.add_argument("--scheduler", type=str, default="noam", choices=['noam', 'linear', 'cyclic', '1cycle','fixedlr'], help="method of optim")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--base_lr", type=float, default=1e-3, help="Initial learning rate which is the lower boundary in the cycle for each parameter group.")
    parser.add_argument("--max_lr", type=float, default=5e-3, help="Upper learning rate boundaries in the cycle for each parameter group.")
    ## CyclicLR
    parser.add_argument("--cycliclr_mode", type=str, default="triangular2", choices=['triangular', 'triangular2', 'exp_range'], help="mode of CyclicLR, see https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR")
   
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
    parser.add_argument("--valid_steps", type=int, default=5000, help="Perfom validation every X steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=2.0, help="Clipping gradient norm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--autocast", action='store_true',
                        help="If true using autocast to automatically mix accuracy to accelerate training(开启自动混合精度加速训练)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")

    args = parser.parse_args()

    # 日志文件配置
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process.
    # logger.info => log main process only, logger.warning => log all processes
    # the name of log file looks like './logs/Jan11_22-55-46_gpu144_GPT_CPED.log'
    # The log information is output to a specified file, which is convenient 
    # for viewing the log when the program is running time specified by the ```at``` command.
    if not os.path.exists(args.log_file):
        # 不存在log目录则创建
        os.makedirs(args.log_file)
    log_file_name_or_tensorboard_dir_name = str(time.strftime('%b%d_%H-%M-%S',time.localtime(time.time())))+'_'+str(socket.gethostname())+'_'+args.model_type+'_'+args.dataset_name
    logging_file_name = os.path.join(args.log_file,log_file_name_or_tensorboard_dir_name+'.log')
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        filename=logging_file_name) # output the log information to the log file 
    logger.warning("Running process %d", args.local_rank)
    logger.info("Arguments: %s", pformat(args))

    # 数据集的训练集、验证集、测试集的文件名称
    cped_filenames = {"train":"train_split.csv",
                      "valid":"valid_split.csv",
                      "test":"test_split.csv"}
    meld_filenames = {"train":"train_sent_emo.csv",
                      "valid":"dev_sent_emo.csv",
                      "test":"test_sent_emo.csv"}
    cped_shuffle_filenames = {"train":"train_shuffle_split.csv",
                              "valid":"valid_shuffle_split.csv",
                              "test":"test_shuffle_split.csv"}
    if args.dataset_name == "MELD":
        filenames = meld_filenames
    elif args.dataset_name == "CPED":
        filenames = cped_filenames
    elif args.dataset_name == "CPED-shuffle":
        filenames = cped_shuffle_filenames

    # 多机分布式训练配置
    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')    

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    
    # 根据模型类型确定模型类以及tokenizer
    if args.model_type == 'GPT':
        model_class = OpenAIGPTLMHeadModel
        config_class = OpenAIGPTConfig
        tokenizer_class = BertTokenizer

    elif args.model_type == 'GPT-2':
        model_class = GPT2LMHeadModel
        config_class = GPT2Config
        tokenizer_class = BertTokenizer    


    '''此处增加模型
    elif args.model_type == 'MODELNAME':
        model_class = ModelClass
        config_class = ModelConfigClass
        tokenizer_class = TokenizerClass
    '''


    # 初始化模型参数    
    if args.pretrained: # 加载预训练模型参数
        tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint, do_lower_case=True)
        model = model_class.from_pretrained(args.model_checkpoint)
    else: # 不是从预训练模型中初始化
        print("不是从预训练模型中初始化")

    # 输出模型结构与参数信息
    if args.show_parameters:
        print(model) # 输出网络结构
        # 输出模型总的参数量
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.modeling_utils.ModuleUtilsMixin.num_parameters
        total_params = model.num_parameters() #直接调用模型自身函数计算模型的参数量
        print(f'{total_params:,} total parameters.')
        
    # 冻结模型的部分层
    if args.freeze_model:
        freeze_by_model_name(model, "transformer.tokens_embed")
        freeze_by_model_name(model, "transformer.positions_embed")
        for i in range(args.freeze_start_layer,args.freeze_end_layer+1):
            # 冻结args.freeze_start_layer~args.freeze_end_layer层
            freeze_by_model_name(model, "transformer.h."+str(i)+".") # 加点在后面，防止冻结1~6时，把第10、11层也冻结了

        # 查看模型的可训练的层
        print("可训练的层如下所示：")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name,':',param.size())
        # 计算模型可训练的参数量
        # total_trainable_params = count_trainable_parameters(model)
        total_trainable_params = model.num_parameters(only_trainable=True)
        print(f'{total_trainable_params:,} total trainable parameters.')

    model.to(args.device)



    if args.L2_regularization: # L2 Regularization
        # 旧版本的L2正则化
        # reference: https://blog.csdn.net/mch2869253130/article/details/105994044
        # 不参与L2正则化的列表
        # optimizer = AdamW([{'params': model.parameters(), 'weight_decay': args.L2_weight_decay, 'initial_lr': args.lr}], lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=args.L2_weight_decay, correct_bias=True)
        
        # 新版本的
        # 参考：https://arxiv.org/pdf/1711.05101.pdf
        # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html?highlight=adamw#torch.optim.AdamW
        # transformers 版本：from transformers import AdamW
        # optimizer = AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, betas=(args.adamw_beta1, args.adamw_beta2), eps=args.adamw_eps, weight_decay=args.L2_weight_decay, correct_bias=True)
        # pytorch版本：from torch.optim import AdamW
        optimizer = AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, betas=(args.adamw_beta1, args.adamw_beta2), eps=args.adamw_eps, weight_decay=args.L2_weight_decay)
        '''
        no_decay = ['bias', 'bias_ih_l0', 'bias_hh_l0', 'LayerNorm.weight','layernorm_1.weight']
        parameters_list = []
        optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.L2_weight_decay, 'initial_lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'initial_lr': args.lr}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=True)
        '''
    else: # not L2 Regularization
        # transformers 版本：from transformers import AdamW
        # optimizer = AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, betas=(args.adamw_beta1, args.adamw_beta2), eps=args.adamw_eps, weight_decay=args.L2_weight_decay, correct_bias=True)
        # pytorch版本：from torch.optim import AdamW
        optimizer = AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, betas=(args.adamw_beta1, args.adamw_beta2), eps=args.adamw_eps)

    if args.autocast:
        # 混合精度训练
        # 参考: https://pytorch.org/docs/1.9.0/amp.html?highlight=torch%20cuda%20amp%20gradscaler
        #       https://pytorch.org/docs/1.9.0/notes/amp_examples.html#amp-examples
        scaler = torch.cuda.amp.GradScaler()  # pytorch版本要求：1.6+

    if args.distributed: 
        # Add "find_unused_parameters=True" to avoid the following error
        # ERROR:ignite.engine.engine.Engine:Current run is terminating due to exception: 
        # Expected to have finished reduction in the prior iteration before starting a new one. 
        # This error indicates that your module has parameters that were not used in producing loss.
        #  You can enable unused parameter detection by (1) passing the keyword argument 
        # `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`;
        if args.find_unused_parameters:
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True) 
        else:
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank) # ,find_unused_parameters=True


    logger.info("Prepare datasets...")
    if args.dataset_name == 'CPED':
        loader_class = build_cped_dataloaders

    if args.model_type == 'GPT' or args.model_type == 'GPT-2':
        train_loader, valid_loader, train_sampler, valid_sampler = build_cped_dataloaders(args, tokenizer, logger, load_test=False, filenames=filenames)
        test_loader, test_sampler = build_cped_dataloaders(args, tokenizer, logger, load_test=True, filenames=filenames)



    # Training function and trainer
    def update(engine, batch):
        model.train()
        # 从batch中读取数据
        if args.model_type == 'GPT' or args.model_type == 'GPT-2':
            input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)


        # 模型进行前向计算
        '''参考代码
        if args.model_type == 'SPEAKERBERT':
            if args.autocast:
                with autocast():
                    (lm_loss), *_ = model(input_ids=input_ids, masked_lm_labels=lm_labels, speaker_type_ids=speaker_type_ids)
            else:
                    (lm_loss), *_ = model(input_ids=input_ids, masked_lm_labels=lm_labels, speaker_type_ids=speaker_type_ids)
        '''
        if args.model_type == 'GPT' or args.model_type == 'GPT-2':
            if args.autocast:
                with autocast():
                    CausalLMOutput = model(input_ids=input_ids, labels=lm_labels, token_type_ids=token_type_ids)
            else:
                CausalLMOutput = model(input_ids=input_ids, labels=lm_labels, token_type_ids=token_type_ids)
            lm_loss = CausalLMOutput.loss
            #print("results=", lm_loss)


        # 反向传递
        if args.autocast:
            with autocast():
                loss = lm_loss / args.gradient_accumulation_steps
        else:
            loss = lm_loss / args.gradient_accumulation_steps
        if args.autocast: # 混合精度训练，要求：torch1.6+
            scaler.scale(loss).backward(retain_graph=args.retain_graph) # retain_graph here is unrelated to amp, it's present because in this both backward() calls share some sections of graph.
            
            if engine.state.iteration % args.gradient_accumulation_steps == 0:
                # 参考：https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
                if args.do_unscale:
                    # 参考：https://pytorch.org/docs/1.9.0/notes/amp_examples.html#amp-examples
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        else:
            loss.backward(retain_graph=args.retain_graph)  # 增加：retain_graph=True，以解决：RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm) # 梯度剪切函数，为防止梯度爆炸
            if engine.state.iteration % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        return loss.item(), optimizer.param_groups[0]['lr']

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        # 从batch中读取数据
        if args.model_type == 'GPT' or args.model_type == 'GPT-2':
            input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)

        with torch.no_grad():
            # 模型进行前向计算
            if args.model_type == 'GPT' or args.model_type == 'GPT-2':
                if args.autocast:
                    with autocast():
                        CausalLMOutput = model(input_ids=input_ids, token_type_ids=token_type_ids)
                else:
                    CausalLMOutput = model(input_ids=input_ids, token_type_ids=token_type_ids)
            
                
                lm_logits = CausalLMOutput.logits
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return lm_logits_flat_shifted, lm_labels_flat_shifted

    evaluator = Engine(inference)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def test(engine, batch):
        model.eval()
        # 从batch中读取数据
        if args.model_type == 'GPT' or args.model_type == 'GPT-2':
            input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels = tuple(None if input_tensor==None else input_tensor.to(args.device) for input_tensor in batch)

        with torch.no_grad():
            # 模型进行前向计算
            if args.model_type == 'GPT' or args.model_type == 'GPT-2':
                if args.autocast:
                    with autocast():
                        CausalLMOutput = model(input_ids=input_ids, token_type_ids=token_type_ids)
                else:
                    CausalLMOutput = model(input_ids=input_ids, token_type_ids=token_type_ids)
                lm_logits = CausalLMOutput.logits
                lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return lm_logits_flat_shifted, lm_labels_flat_shifted

    testor = Engine(test)


    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(valid_loader))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: testor.run(test_loader))

    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(valid_loader))
        trainer.add_event_handler(Events.COMPLETED, lambda _: testor.run(test_loader))

    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(valid_loader))
        trainer.add_event_handler(Events.COMPLETED, lambda _: testor.run(test_loader))


    # Evaluation during training
    @trainer.on(Events.ITERATION_STARTED)
    def log_iterations(engine):
        # if engine.state.iteration % max(int(0.1 * len(train_loader)), 1) == 0:
        if engine.state.iteration % args.valid_steps == 0:
            evaluator.run(valid_loader)
            testor.run(test_loader)

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))
        testor.add_event_handler(Events.EPOCH_STARTED, lambda engine: test_sampler.set_epoch(engine.state.epoch))

    # noam decrease the learning rate
    # d_model = model.config.n_embd
    # 参考论文《transformer is all you need》第5.3节
    d_model = args.n_emd
    noam_lambda = lambda step: (
            d_model ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5)))
    noam_scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda, last_epoch=args.from_step)
    if args.scheduler == "noam":
        scheduler = LRScheduler(noam_scheduler)
    if args.scheduler == "linear":
        scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    if args.scheduler == "cyclic":
        # 文章https://arxiv.org/pdf/1506.01186.pdf
        scheduler = LRScheduler(CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.max_lr, step_size_up=2000, step_size_down=2000, mode=args.cycliclr_mode, gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1))
    if args.scheduler == "1cycle":
        scheduler = LRScheduler(OneCycleLR(optimizer, args.lr, total_steps=args.n_epochs, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=False, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1))
    if args.scheduler == "fixedlr":
        scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), args.lr)])

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)



    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "lr")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0], x[1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
        metric.attach(testor, name)

    # On the main process: add progress bar, tensorboard, checkpoints
    # And save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True, mininterval=2)
        pbar.attach(trainer, metric_names=["loss", "lr"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))
        testor.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Test: %s" % pformat(testor.state.metrics)))

        # tb_logger = TensorboardLogger(log_dir=None, comment='_'+args.model_type+'_'+args.dataset_name)
        # 统一日志名称和tensorboard输出文件夹名称, 2022.04.11
        tb_logger = TensorboardLogger(log_dir=os.path.join("./runs/",log_file_name_or_tensorboard_dir_name))

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)

        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        
        tb_logger.attach_output_handler(evaluator,
                                        event_name=Events.EPOCH_COMPLETED,
                                        tag="validation",
                                        metric_names=list(metrics.keys()),
                                        global_step_transform=global_step_from_engine(trainer))
        tb_logger.attach_output_handler(testor,
                                        event_name=Events.EPOCH_COMPLETED,
                                        tag="test",
                                        metric_names=list(metrics.keys()),
                                        global_step_transform=global_step_from_engine(trainer))
        '''
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())),
                         event_name=Events.EPOCH_COMPLETED, global_step_transform=global_step_from_engine(trainer))
        tb_logger.attach(testor, log_handler=OutputHandler(tag="test", metric_names=list(metrics.keys())),
                         event_name=Events.EPOCH_COMPLETED, global_step_transform=global_step_from_engine(trainer))
        '''

        # 连续3个epoch，测试集的ppl没有下降就停止训练
        early_stop_handler = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)
        testor.add_event_handler(Events.COMPLETED, early_stop_handler)

        #checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
        # ignite v0.4.8版本

        best_model_handler = Checkpoint(
                                        {"model": model},
                                        tb_logger.writer.log_dir,
                                        filename_prefix="best",
                                        n_saved=2,
                                        global_step_transform=global_step_from_engine(trainer),
                                        score_name="test_ppl",
                                        score_function=score_function,
                                    )
        testor.add_event_handler(Events.COMPLETED, best_model_handler)

        '''
        # save model after evaluation
        testor.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation
        '''

        torch.save(args, tb_logger.writer.log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))

        #tokenizer.save_vocabulary(tb_logger.writer.log_dir)
        # save the new tokens vacab
        tokenizer.save_pretrained(tb_logger.writer.log_dir)
        with open(tb_logger.writer.log_dir + "/training_args.json",'w',encoding='utf-8') as json_file:
            json.dump(pformat(args),json_file,ensure_ascii=False)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint
    # (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        # 重命名模型名称为WEIGHTS_NAME指定的字符串
        # ignite0.4.8有所调整
        # os.rename(os.path.join(tb_logger.writer.log_dir, checkpoint_handler._saved[-1][1]),
        #          os.path.join(tb_logger.writer.log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        os.rename(os.path.join(tb_logger.writer.log_dir, best_model_handler.last_checkpoint),
                  os.path.join(tb_logger.writer.log_dir, WEIGHTS_NAME))
        
        tb_logger.close()


if __name__ == "__main__":
    train()