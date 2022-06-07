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
model_checkpoint=~/PretrainedModel/CDial-GPT/CDial-GPT_LCCC-base
cache_path=../data/CPED_cache_for_CpedDataset

# GPT
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 2301 train_model.py --model_type=GPT --model_checkpoint=~/PretrainedModel/CDial-GPT/CDial-GPT_LCCC-base --pretrained --dataset_name=CPED --data_path=../data/CPED/ --cache_path=../data/CPED_cache_for_CpedDataset --lr=6.25e-5 --scheduler=noam --autocast --train_batch_size=1 --valid_batch_size=1 --test_batch_size=1 --n_epochs=2
