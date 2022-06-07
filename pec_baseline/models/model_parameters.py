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

# Model parameters processing file
# File: model_parameters.py
# Used for model parameters analysis
# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2022.04.06

def count_trainable_parameters(model):
    '''获取需要训练的参数数量
    使用示例：print(f'The model has {count_trainable_parameters(model):,} trainable parameters')
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

def count_total_parameters(model):
    '''获取模型总的参数数量
    使用示例：print(f'The model has {count_total_parameters(model):,} total parameters')
    '''
    return sum(p.numel() for p in model.parameters()) 

def show_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

# 冻结模型参数
# 参考：https://blog.csdn.net/weixin_41712499/article/details/111295683?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-5.no_search_link&spm=1001.2101.3001.4242
def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)
 
def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)
 
def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)
 
def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)

def freeze_by_model_name(model, model_name):
    for name, param in model.named_parameters():
        if name.startswith(model_name):
            param.requires_grad = False

def unfreeze_by_model_name(model, model_name):
    for name, param in model.named_parameters():
        if name.startswith(model_name):
            param.requires_grad = True