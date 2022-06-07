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

# Dataset data loading file
# File: base_util.py
# Used for dataset loading
# 用于数据集读取的基础方法
# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2022.03.21

import os
import re
import math
import json
import shutil
import random
import collections
import importlib.util
import pandas as pd
import logging
from io import open
from os.path import join
from .dataset_statistics import get_row_statistics

logger = logging.getLogger(__name__)


_torch_available = importlib.util.find_spec("torch") is not None


def is_torch_available():
    return _torch_available

def load_csv_data_from_dir(data_dir="../data/CPED",
                           file_dict={"train":"train_split.csv",
                                      "valid":"valid_split.csv",
                                      "test":"test_split.csv"}):
    '''get_data from dir, which have train_split.csv, valid_split.csv, test_split.csv file
    Inputs:
        **data_dir**: str, 

    '''
    print("Read dataset from ", data_dir)
    train_data = pd.read_csv(join(data_dir,file_dict["train"]), encoding="UTF-8-SIG")
    valid_data = pd.read_csv(join(data_dir,file_dict["valid"]), encoding="UTF-8-SIG")
    test_data = pd.read_csv(join(data_dir,file_dict["test"]), encoding="UTF-8-SIG")
    return train_data, valid_data, test_data


def shuffle_total_data(data_path, 
                       save_path, 
                       validation_split_percentage=0.1, 
                       test_split_percentage=0.1, 
                       file_names = ["train_shuffle_split.csv", "valid_shuffle_split.csv", "test_shuffle_split.csv"],
                       regen=False):
    '''shuffle_total_data
    功能：将一个.csv文件随机打乱，拆分为训练集、验证集、测试集，分别保存
    输入：
    data_path: .csv文件的路径
    save_path: 拆分后的文件保存的目录
    validation_split_percentage: 验证集比例
    
    '''
    if regen==False:
        print("不进行重复生成！")
        return False
    else:
        # 以下操作删除原先的文件，为危险操作
        if os.path.exists(join(save_path,file_names[0])):
            os.remove(join(save_path,file_names[0]))

        if os.path.exists(join(save_path,file_names[1])):
            os.remove(join(save_path,file_names[1]))
            
        if os.path.exists(join(save_path,file_names[2])):
            os.remove(join(save_path,file_names[2]))

        print("Read dataset from ", data_path)
        data = pd.read_csv(data_path, 
                        usecols=["Dialogue_ID","Utterance_ID","Speaker","Sentiment","Emotion","DA","Utterance","Gender","Age","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"], 
                        encoding="UTF-8-SIG")
        # 划分为训练集、测试集
        keys = list(set(data['Dialogue_ID']))
        random.shuffle(keys) # 随机打乱
        validation_split_id = int(len(keys)*(1-validation_split_percentage-test_split_percentage))
        test_split_id = int(len(keys)*(1-test_split_percentage))
        train_keys = keys[:validation_split_id] # 训练集索引
        valid_keys = keys[validation_split_id:test_split_id] # 验证集索引
        test_keys = keys[test_split_id:] # 测试集索引
        train_data = data[data['Dialogue_ID'].isin(train_keys)]
        valid_data = data[data['Dialogue_ID'].isin(valid_keys)]
        test_data = data[data['Dialogue_ID'].isin(test_keys)]
        
        train_data.to_csv(join(save_path,file_names[0]), encoding="UTF-8-SIG", index=False)
        valid_data.to_csv(join(save_path,file_names[1]), encoding="UTF-8-SIG", index=False)
        test_data.to_csv(join(save_path,file_names[2]), encoding="UTF-8-SIG", index=False)
        print("已经完成数据集生成！")

        return True


# 将指定路径下的所有csv文件合并为一个csv文件
# 实现该函数主要方便汇总统计
def combine_csv_files(data_path="./MELD/", 
                      save_path="./MELD/MELD_total_text.csv",
                      regen=False):
    '''combine_csv_files
    将指定路径下的所有csv文件合并为一个csv文件

    使用示例：
    
    combine_csv_files(data_path="./MELD/", 
                      save_name="MELD_total_text", 
                      files=file_names, 
                      save_in="./MELD/")

    '''
    if regen==False:
        print("不进行重复生成！")
        return False



    files = os.listdir("./MELD/")
    if os.path.isfile(join(save_in, "%s.csv") % save_name):
        return 0
    else:
        try:
            main_list = []
            for i in range(len(files)):
                content = pd.read_csv(join(data_path, files[i]), encoding="UTF-8-SIG")
                if i == 0:
                    main_list.extend([list(content.keys())])
                main_list.extend(content.values.tolist())

            main_dict = {}
            for i in list(zip(*main_list)):
                main_dict[i[0]] = list(i[1:])
            data_df = pd.DataFrame(main_dict)
            data_df.to_csv(join(save_in, "%s.csv") % save_name, encoding="UTF-8-SIG", index=False)
        except:
            print("合并[%s]时发生错误" % save_name)



def save_speaker(data_path, save_path, row_name="Speaker", regen=False):
    '''读取数据集的所有姓名，制作姓名表
       使用示例：
       save_speaker(data_path="/148Dataset/Dataset/MELD/MELD/train_sent_emo.csv", 
                    save_path="/148Dataset/Dataset/MELD/MELD/speakers.txt", 
                    row_name="Speaker", 
                    regen=True)
    '''
    if os.path.exists(save_path):
        if regen == False:
            return None
        elif regen == True:
            os.remove(save_path)

    data = pd.read_csv(data_path, encoding="UTF-8-SIG")
    results = get_row_statistics(data,row_name)
    print(results["keys"])
    print(results["element_stastics"])
    with open(save_path, 'w') as f:
        for i in range(len(results["keys"])):
            f.write(results["keys"][i]+'\n')
    return True


def load_speaker(speakers_file): #speakers.txt
    """Loads a speaker file into a dictionary.
       speakers_file: 姓名汇总表格
       speakers: 返回的列表形式的表格
       speakers_to_ids: 返回的字典，通过该字典可以根据姓名获得对应的id
       ids_to_speakers：通过该字典可以根据id获得对应的姓名
    """
    speakers_to_ids = collections.OrderedDict()
    with open(speakers_file, "r", encoding="utf-8") as reader:
        speakers = reader.readlines()
    for index, token in enumerate(speakers):
        token = token.rstrip('\n')
        speakers[index] = token
        speakers_to_ids[token] = index
    ids_to_speakers = collections.OrderedDict([(ids, tok) for tok, ids in speakers_to_ids.items()])
    return speakers, speakers_to_ids, ids_to_speakers


def convert_speaker_to_id(speakers_to_ids, speaker, unk_token="其他"):
    """ Converts a speaker (str/unicode) in an id using the speakers. """
    return speakers_to_ids.get(speaker, speakers_to_ids.get(unk_token))

def convert_id_to_speaker(ids_to_speakers, index, unk_token="其他"):
    """Converts an index (integer) in a speaker (string/unicode) using the speakers."""
    return ids_to_speakers.get(index, unk_token)

def convert_cache_to_csv(dataset_cache,output_dir):
    
    data = torch.load(dataset_cache)
    train_data = data["train"]
    valid_data = data["valid"]
    test_data = data["test"]
    train_data.to_csv(join(output_dir,dataset_cache+"train.csv"),columns=["Dialogue_ID","Utterance_ID","Speaker","Sentiment","Emotion","DA","Utterance","Gender","Age","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"])

