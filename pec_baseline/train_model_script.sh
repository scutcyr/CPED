#!/bin/bash
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

# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2022.04.06

# Train model with torch.distributed.launch by using torch.distributed.launch
# Trained on Ubuntu18.04 with 2 GeForce RTX 2080ti GPUs and 125G Memory
# --n_epochs=120 --warmup_steps=10000 --lr=5.5e-3 --pretrained --train_batch_size=4 --valid_batch_size=2 --test_batch_size=1 --num_workers=4

# CPU个数：2，单个CPU核心数：10
# cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l
# cat /proc/cpuinfo| grep "cpu cores"| uniq



#############################################################################
# 中文数据集CPED
dataset_name=CPED
data_path=../data/CPED
model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CDial-GPT_LCCC-base
cache_path=../data/CPED_cache_for_CpedDataset

# GPT

CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 2301 train_model.py --model_type=GPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CDial-GPT_LCCC-base --pretrained --dataset_name=CPED --data_path=../data/CPED/ --cache_path=../data/CPED_cache_for_CpedDataset --lr=6.25e-5 --scheduler=noam --autocast --train_batch_size=1 --valid_batch_size=1 --test_batch_size=1 --n_epochs=2

CUDA_VISIBLE_DEVICES=2,5 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 2301 train_model.py --model_type=GPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CDial-GPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --lr=5.5e-3 --scheduler=noam --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=1 --valid_batch_size=1 --test_batch_size=1 --n_epochs=64


# CVGPT

CUDA_VISIBLE_DEVICES=2,5 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 2311 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --lr=5.5e-3 --scheduler=noam --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=1 --valid_batch_size=1 --test_batch_size=1 --n_epochs=64  --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da

#
CUDA_VISIBLE_DEVICES=1,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 2312 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --lr=6.25e-5 --scheduler=linear --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32  --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da


# Apr08_19-31-02_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 2313 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --lr=6.25e-5 --scheduler=noam --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da

# 2022.04.09 03:15 Apr09_03-21-14_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 2314 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --lr=6.25e-4 --scheduler=noam --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da

# 2022.04.09 03:45 Apr09_03-49-22_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12318 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --lr=6.25e-3 --scheduler=noam --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da

# 以下部分代码开始增加--do_unscale、--retain_graph、--adamw_beta1、--adamw_beta2、--adamw_eps
# 2022.04.09 05:15 Apr09_05-19-04_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12319 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --lr=6.25e-3 --scheduler=noam --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 06:00 Apr09_06-04-06_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12320 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --lr=1e-3 --scheduler=noam --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 07:15 Apr09_07-19-03_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12321 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --lr=1e-3 --scheduler=noam --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 08:00 Apr09_08-04-29_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12322 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --lr=1e-3 --scheduler=noam --L2_regularization --L2_weight_decay=1e-1 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 09:15 Apr09_09-19-31_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12323 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=cyclic --base_lr=1e-3 --max_lr=5e-3  --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 10:00 Apr09_10-03-57_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12324 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=cyclic --cycliclr_mode=triangular --base_lr=1e-7 --max_lr=5e-7  --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 11:15 Apr09_11-18-57_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12325 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=cyclic --cycliclr_mode=triangular --base_lr=1e-7 --max_lr=1e-6  --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph


# 2022.04.09 12:00 Apr09_12-03-50_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12326 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=fixedlr --lr=5e-7  --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 13:15 Apr09_13-19-14_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12327 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=fixedlr --lr=5e-6  --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 14:00 Apr09_14-04-45_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12328 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=fixedlr --lr=5e-5  --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 15:15 Apr09_15-19-47_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12329 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=fixedlr --lr=5e-7  --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 16:00 Apr09_16-05-23_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12330 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=fixedlr --lr=5e-7  --L2_regularization --L2_weight_decay=1e-1 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph


# 2022.04.09 17:15 Apr09_17-19-46_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12331 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --lr=6.25e-4 --scheduler=noam --L2_regularization --L2_weight_decay=5e-1 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph


# 2022.04.09 18:00 Apr09_18-05-04_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12332 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=cyclic --cycliclr_mode=triangular2 --base_lr=1e-8 --max_lr=6.25e-5 --L2_regularization --L2_weight_decay=1e-1 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=64 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 19:00 
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12333 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=cyclic --cycliclr_mode=triangular2 --base_lr=1e-8 --max_lr=6.25e-6 --L2_regularization --L2_weight_decay=1e-1 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=64 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 2022.04.09 22:10 Apr09_22-14-12_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12334 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=fixedlr --lr=6.25e-7  --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=16 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph

# 一个epoch，查看模型及参数 Apr09_22-17-49_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12335 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=fixedlr --lr=6.25e-7  --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=1 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters

# 2022.04.09 23:10 冻结GPT的前8层，只保留后4层
# Apr09_23-27-34_gpu144_CVGPT_CPED，下降太慢
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12336 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=6.25e-5 --L2_regularization --L2_weight_decay=1e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --freeze_model --freeze_start_layer=0 --freeze_end_layer=7 --show_parameters

# Apr09_23-43-11_gpu144_CVGPT_CPED，过拟合
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12337 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --freeze_model --freeze_start_layer=0 --freeze_end_layer=7 --show_parameters

# Apr10_00-58-38_gpu144_CVGPT_CPED，下降太慢
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12338 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=6.25e-5 --L2_regularization --L2_weight_decay=1e-1 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --freeze_model --freeze_start_layer=0 --freeze_end_layer=7 --show_parameters

# Apr10_01-26-32_gpu144_CVGPT_CPED，过拟合
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12339 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=6 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=5000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --freeze_model --freeze_start_layer=0 --freeze_end_layer=3 --show_parameters

# 不进行混合精度训练，参考先前的CPED的./runs/Jun11_14-50-58_gpu144_GPT_EMO_PER的参数配置
# 2022.04.10 03:15 Apr10_03-15-44_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.1 --master_port 12340 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --train_batch_size=4 --valid_batch_size=1 --test_batch_size=1 --n_epochs=120 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --gradient_accumulation_steps=64

# Apr10_04-57-05_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,2,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.1 --master_port 12341 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --freeze_model --freeze_start_layer=0 --freeze_end_layer=9 --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

CUDA_VISIBLE_DEVICES=1,2,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.1 --master_port 12342 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=4 --valid_batch_size=1 --test_batch_size=1 --n_epochs=120 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --gradient_accumulation_steps=64 --not_return_dict


CUDA_VISIBLE_DEVICES=1,2,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.1 --master_port 12343 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=64 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --freeze_model --freeze_start_layer=0 --freeze_end_layer=9 --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict


CUDA_VISIBLE_DEVICES=1,2,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.1 --master_port 12344 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=64 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --freeze_model --freeze_start_layer=0 --freeze_end_layer=10 --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict


CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12345 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=64 --warmup_steps=10000 --set_current_speaker_mask --do_unscale --retain_graph --freeze_model --freeze_start_layer=0 --freeze_end_layer=9 --show_parameters --max_history=16 --max_norm=1.0

# 重点关注：Apr11_01-17-46_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12346 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=6.25e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=4 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12347 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=6.25e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=4 --valid_batch_size=1 --test_batch_size=1 --n_epochs=16 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# 2022.04.11 测试保存best模型 Apr11_05-52-42_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12348 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=6.25e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=4 --valid_batch_size=1 --test_batch_size=1 --n_epochs=16 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12349 train_model.py --model_type=CVGPT --model_checkpoint=./runs/Apr11_05-52-42_gpu144_CVGPT_CPED --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=6.25e-4 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=4 --valid_batch_size=1 --test_batch_size=1 --n_epochs=16 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict


CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12350 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=4 --valid_batch_size=2 --test_batch_size=1 --n_epochs=120 --warmup_steps=5000 --set_current_speaker_mask --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0

















####################################
# CCVGPT
# 比较不同的对话历史长度（对话轮数）窗口, 2022.04.12 01:30

# --max_history=4 Apr12_01-22-56_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12351 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=4 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=4 --max_norm=1.0 --not_return_dict

# --max_history=8 Apr12_01-24-20_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12352 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=4 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=8 --max_norm=1.0 --not_return_dict

# --max_history=12 Apr12_01-36-14_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12353 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=4 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=12 --max_norm=1.0 --not_return_dict

# --max_history=16 Apr12_01-52-31_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12354 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# --max_history=20 
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12355 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=20 --max_norm=1.0 --not_return_dict

# --max_history=24 Apr12_01-53-13_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12356 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=24 --max_norm=1.0 --not_return_dict

####################################
# 比较是否进行混合精度训练
# 进行混合精度训练：Apr12_01-52-31_gpu144_CVGPT_CPED

# 不进行混合精度训练，去掉 --autocast, Apr12_04-55-42_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12361 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict


####################################
#
# 消融实验
# Apr12_01-52-31_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12354 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# 使用from torch.optim import AdamW替换原来的from transformers import AdamW
# 去掉--with_current_da，2022.04.15 22:31
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12401 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# 去掉--with_current_emotion，2022.04.15 22:31
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12402 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# 去掉--with_current_persona，2022.04.15 22:31
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12403 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_da --with_current_emotion --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict


####################################
# 比较冻结不同的GPT层数, 2022.04.12

# 冻结0层：Apr12_01-52-31_gpu144_CVGPT_CPED，效果最好

# 冻结前4层 Apr12_04-46-44_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12357 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --freeze_model --freeze_start_layer=0 --freeze_end_layer=3

# 冻结前6层 Apr12_04-47-18_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12358 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --freeze_model --freeze_start_layer=0 --freeze_end_layer=5

# 冻结前8层 Apr12_04-47-52_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12359 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --freeze_model --freeze_start_layer=0 --freeze_end_layer=7

# 冻结前10层 Apr12_04-52-03_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12360 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --freeze_model --freeze_start_layer=0 --freeze_end_layer=10


####################################
# 比较不同的学习率模式

# noam --lr=6e-3, 
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12365 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=6e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# noam --lr=5.5e-3, Apr12_01-52-31_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12366 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# noam --lr=5e-3, Apr12_10-59-01_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12367 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# noam --lr=1e-3, Apr12_10-59-43_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12368 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=1e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# noam --lr=5e-4, Apr12_11-00-58_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12369 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5e-4 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# noam --lr=1e-4, Apr12_11-10-01_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12370 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=1e-4 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# noam --lr=6e-3, Apr12_11-22-20_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12371 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=6e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# noam --lr=2e-1, Apr12_12-46-57_gpu144_CVGPT_CPED
# 参考：https://www.cvmart.net/community/detail/4031
# 训练Transformer会需要比0.001更大的初始学习率，默认设置是学习率0.2+NOAM Scheduler。调节学习率对结果影响很大，可以说是优化器最重要的超参数。
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12375 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=2e-1 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# noam --lr=3e-4, Apr12_12-56-41_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12376 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=3e-4 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# 固定学习率 , 2022.06.04 01:35 Jun04_01-35-24_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12381 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=fixedlr --lr=6.25e-5 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=1 --valid_batch_size=1 --test_batch_size=1 --n_epochs=32 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --adamw_beta1=0.9 --adamw_beta2=0.98 --adamw_eps=1e-9 --alpha_nll=1.0 --alpha_emotion=0.1 --alpha_da=0.1 --alpha_per_gen=0.1 --alpha_per_neu=0.1 --alpha_per_ext=0.1 --alpha_per_ope=0.1 --alpha_per_agr=0.1 --alpha_per_con=0.1

# CyclicLR, 2022.06.04 02:48
# --scheduler=cyclic --cycliclr_mode=triangular2 --base_lr=1e-6 --max_lr=6.25e-3
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 12382 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=cyclic --cycliclr_mode=triangular2 --base_lr=1e-6 --max_lr=6.25e-3 --lr=6.25e-5 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=1 --valid_batch_size=1 --test_batch_size=1 --n_epochs=64 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --adamw_beta1=0.9 --adamw_beta2=0.98 --adamw_eps=1e-9 --alpha_nll=1.0 --alpha_emotion=0.1 --alpha_da=0.1 --alpha_per_gen=0.1 --alpha_per_neu=0.1 --alpha_per_ext=0.1 --alpha_per_ope=0.1 --alpha_per_agr=0.1 --alpha_per_con=0.1


####################################
# 比较不同的优化器超参数设置，beta1、beta2、eps

# --adamw_beta1=0.9 --adamw_beta2=0.999 --adamw_eps=1e-6
# Apr12_01-52-31_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12354 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# --adamw_beta1=0.9 --adamw_beta2=0.999 --adamw_eps=1e-8
# Apr12_11-37-00_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12373 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --adamw_beta1=0.9 --adamw_beta2=0.999 --adamw_eps=1e-8

# --adamw_beta1=0.9 --adamw_beta2=0.999 --adamw_eps=1e-9
# Apr12_13-13-04_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12377 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --adamw_beta1=0.9 --adamw_beta2=0.999 --adamw_eps=1e-9


# --adamw_beta1=0.9 --adamw_beta2=0.98 --adamw_eps=1e-9
# Apr12_13-14-19_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12378 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --adamw_beta1=0.9 --adamw_beta2=0.98 --adamw_eps=1e-9




####################################
# 比较不同的L2正则化系数, # --adamw_beta1=0.9 --adamw_beta2=0.98 --adamw_eps=1e-9
# --L2_weight_decay=5e-2 Apr12_13-14-19_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12378 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --adamw_beta1=0.9 --adamw_beta2=0.98 --adamw_eps=1e-9

# --L2_weight_decay=1e-1 
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12379 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=1e-1 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --adamw_beta1=0.9 --adamw_beta2=0.98 --adamw_eps=1e-9

# --L2_weight_decay=2e-1 
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12380 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=2e-1 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --adamw_beta1=0.9 --adamw_beta2=0.98 --adamw_eps=1e-9



####################################
# 比较不同的loss占比，alpha1~alpha8
# loss = alpha1*loss_nll+alpha2*loss_pre_neu+alpha3*loss_pre_ext+alpha4*loss_pre_ope+alpha5*loss_pre_agr+alpha6*loss_pre_con+alpha7*loss_pre_da+alpha8*loss_pre_emo
# 各个系数均为1.0 Apr12_01-52-31_gpu144_CVGPT_CPED
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12354 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict

# 2022.04.15 23:12
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12501 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --alpha_nll=1.0 --alpha_emotion=1.0 --alpha_da=1.0 --alpha_per_gen=0.5 --alpha_per_neu=0.5 --alpha_per_ext=0.5 --alpha_per_ope=0.5 --alpha_per_agr=0.5 --alpha_per_con=0.5

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12502 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --alpha_nll=1.0 --alpha_emotion=0.5 --alpha_da=1.0 --alpha_per_gen=1.0 --alpha_per_neu=1.0 --alpha_per_ext=1.0 --alpha_per_ope=1.0 --alpha_per_agr=1.0 --alpha_per_con=1.0

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12503 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --alpha_nll=1.0 --alpha_emotion=1.0 --alpha_da=0.5 --alpha_per_gen=1.0 --alpha_per_neu=1.0 --alpha_per_ext=1.0 --alpha_per_ope=1.0 --alpha_per_agr=1.0 --alpha_per_con=1.0

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12504 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --alpha_nll=1.0 --alpha_emotion=0.5 --alpha_da=0.5 --alpha_per_gen=1.0 --alpha_per_neu=1.0 --alpha_per_ext=1.0 --alpha_per_ope=1.0 --alpha_per_agr=1.0 --alpha_per_con=1.0

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12505 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --alpha_nll=1.0 --alpha_emotion=0.5 --alpha_da=0.5 --alpha_per_gen=0.5 --alpha_per_neu=0.5 --alpha_per_ext=0.5 --alpha_per_ope=0.5 --alpha_per_agr=0.5 --alpha_per_con=0.5

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12506 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --alpha_nll=1.0 --alpha_emotion=0.5 --alpha_da=0.5 --alpha_per_gen=0.1 --alpha_per_neu=0.1 --alpha_per_ext=0.1 --alpha_per_ope=0.1 --alpha_per_agr=0.1 --alpha_per_con=0.1

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12507 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --alpha_nll=1.0 --alpha_emotion=0.1 --alpha_da=0.1 --alpha_per_gen=0.1 --alpha_per_neu=0.1 --alpha_per_ext=0.1 --alpha_per_ope=0.1 --alpha_per_agr=0.1 --alpha_per_con=0.1

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 12508 train_model.py --model_type=CVGPT --model_checkpoint=/home/phd-chen.yirong/PretrainedModel/CDial-GPT/CVGPT_LCCC-base --pretrained --dataset_name=CPED --data_path=./data/CPED/ --cache_path=./data/CPED_cache_for_CpedDataset --scheduler=noam --lr=5.5e-3 --L2_regularization --L2_weight_decay=5e-2 --autocast --train_batch_size=2 --valid_batch_size=1 --test_batch_size=1 --n_epochs=160 --warmup_steps=10000 --set_current_speaker_mask --with_current_persona --with_current_emotion --with_current_da --do_unscale --retain_graph --show_parameters --max_history=16 --max_norm=1.0 --not_return_dict --alpha_nll=1.0 --alpha_emotion=0.2 --alpha_da=0.2 --alpha_per_gen=0.2 --alpha_per_neu=0.2 --alpha_per_ext=0.2 --alpha_per_ope=0.2 --alpha_per_agr=0.2 --alpha_per_con=0.2





