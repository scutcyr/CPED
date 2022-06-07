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

# Dataset statistic file
# File: dataset_statistics.py
# Used for dataset analysis
# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2022.03.21

import os
import numpy as np
import pandas as pd
from os.path import join
from collections import Counter


def get_data_for_analysis(data_dir="../data/CPED",
                          file_dict={"train":"train_split.csv","valid":"valid_split.csv","test":"test_split.csv"}):
    '''get_data from dir, which have train_split.csv, valid_split.csv, test_split.csv file
    Get .csv format dataset from data_dir.
    '''
    print("Read dataset from ", data_dir)
    train_data = pd.read_csv(join(data_dir,file_dict["train"]), encoding="UTF-8-SIG")
    valid_data = pd.read_csv(join(data_dir,file_dict["valid"]), encoding="UTF-8-SIG")
    test_data = pd.read_csv(join(data_dir,file_dict["test"]), encoding="UTF-8-SIG")
    return train_data, valid_data, test_data


def get_totaldata_for_analysis(data_path="/home/MMMTD/data/processed_cleaned_data/total_checked_processed_cleaned_data/checked_processed_cleaned_data.csv"):
    '''get total data from data_path
    Get .csv format dataset from data_path.
    '''
    print("Read dataset from ", data_path)
    total_data = pd.read_csv(data_path, encoding="UTF-8-SIG")
    return total_data


def get_row_statistics(data,row_name):
    '''get_row_statistics
    Get dataset row statistics with row_name
    E.g.
    train_data_TV_ID=get_row_statistics(train_data,"TV_ID")
    print("train TV stastics:\n", train_data_TV_ID["element_stastics"])
    '''
    name = row_name
    keys = list(set(data[name]))
    values = data[name].tolist()
    element_stastics=pd.value_counts(values)
    
    row_size = len(values)
    row_class = len(keys)
    results={"name":name,"keys":keys,"values":values,"element_stastics":element_stastics,"size":row_size,"class":row_class}
    return results


def cout_dialogue_words(data,dialogue_id):
    dialogue_data = data[data['Dialogue_ID']==dialogue_id]
    count = 0
    for utt in dialogue_data["Utterance"]:
        count = count + len(str(utt))
    return count



def remove_element(utt_list,word = " "):
    temp_list = utt_list
    while word in temp_list:
        temp_list.remove(word)
    return temp_list


def statistics_utterance(data,row_name="Utterance"):
    '''statistics_utterance
    Count the average and maximum word count of sentences

    '''
    utt_list = data[row_name].tolist()
    utt_word_list = [remove_element(list(utterance)) for utterance in utt_list]
    count_utt_word = []
    for utt in utt_word_list:
        count_utt_word.append(len(utt))
    return {"max":max(count_utt_word),"avg":sum(count_utt_word)/len(count_utt_word)}


def statistics_emotda(data,row_name="Emotion",dialogue_id="Dialogue_ID"):
    '''statistics_emotda
    Count the average emotion/DA per Dialogue

    '''
    keys = list(set(data[dialogue_id]))
    count_dial_eda = []
    for key in keys:
        dial_eda = list(set(data[data[dialogue_id]==key][row_name].tolist()))
        count_dial_eda.append(len(dial_eda))
    return {"max":max(count_dial_eda),"avg":sum(count_dial_eda)/len(count_dial_eda)}


def statistics_avg_duration(all_data,data,dialogue_id="Dialogue_ID",StartTime="StartTime",EndTime="EndTime"):
    '''statistics_emotda
    Count the average emotion/DA per Dialogue

    '''
    keys = list(set(data[dialogue_id]))
    count_dial_time = []
    for key in keys:
        start_time = all_data[all_data[dialogue_id]==key][StartTime]
        end_time = all_data[all_data[dialogue_id]==key][EndTime]
        time_list = np.array([int(s) for s in end_time.tolist()])-np.array([int(s) for s in start_time.tolist()])
        time_list = time_list.tolist()

        count_dial_time.append(sum(time_list)/len(time_list))
    return {"avg":sum(count_dial_time)/len(count_dial_time)}


def print_speaker(data_path):
    print("Output all the name of speakers from "+data_path)
    data = get_totaldata_for_analysis(data_path)
    data_speaker_result = get_row_statistics(data,"说话者姓名")
    print(data_speaker_result["keys"])
    print(data_speaker_result["class"])


def print_speaker_from_dir(input_dir="/home/MMMTD/data/processed_cleaned_data/checked_processed_cleaned_data"):
    if not os.path.exists(input_dir):
        print("unexisted data dir："+input_dir)
        return False
    print("From"+input_dir+"load file......")
    file_names = os.listdir(input_dir)
    for file_name in file_names:
        print_speaker(data_path= join(input_dir,file_name))

    return True


def statistic_speaker(data_path = "/home/MMMTD/data/MMMTD_cleaned_speaker_annotation.xlsx"):
    if not os.path.isfile(data_path):
        print("unexisted data dir："+input_dir)
        return 0
    else:
        data = pd.read_excel(data_path)  # xlrd==1.2.0, do not use xlrd==2.0.1
        gender_result = get_row_statistics(data,"性别")
        age_result = get_row_statistics(data,"年龄段")
    return gender_result, age_result


def print_sentiment(data_path, sentiment='中性情绪'):
    print("From "+data_path+" return "+sentiment)
    data = get_totaldata_for_analysis(data_path)
    data_result = get_row_statistics(data,"情绪(粗粒度)")
    print(data_result['element_stastics'][sentiment])
    return data_result['element_stastics'][sentiment]/data_result['size']


def print_sentiment_from_dir(input_dir="/home/MMMTD/data/processed_cleaned_data/checked_processed_cleaned_data"):
    if not os.path.exists(input_dir):
        print("unexisted data dir："+input_dir)
        return False
    print("From"+input_dir+"load file......")
    file_names = os.listdir(input_dir)
    result_dir = {}
    for file_name in file_names:
        result_dir[file_name] = print_sentiment(data_path= join(input_dir,file_name), sentiment='中性情绪')

    return result_dir


def count_eda_array_from_data(data,da_name = "DA", emotion_name = "Emotion", utt_id_name = "Utterance_ID"):
    # Get DA and Emotion labels
    da_label = list(set(data[da_name].tolist()))
    emotion_label = list(set(data[emotion_name].tolist()))
    print("da_label=",da_label)
    print("da_label数目", len(da_label))
    print("emotion_label=",emotion_label)
    print("emotion_label数目", len(emotion_label))
    # add index to label
    da_id = {}
    id = 0
    for da in da_label:
        da_id[da]=id
        id = id+1
    print(da_id)

    emotion_id={}
    id = 0
    for emotion in emotion_label:
        emotion_id[emotion]=id
        id = id+1
    print(emotion_id)

    #count_eda_array = np.zeros((len(da_label),len(emotion_label)))  
    count_eda_array = np.zeros((len(emotion_label),len(da_label)))  
    # output the initial array
    print(count_eda_array)

    # begin statistics

    for utt_id in data[utt_id_name].tolist():
        # for the emotion and DA of each row
        # print(data[data[utt_id_name]== utt_id ][da_name])
        current_da_id = da_id[ str(data[data[utt_id_name]== utt_id ][da_name].values.astype("str")[0]) ]
        current_emotion_id = emotion_id[ str(data[data[utt_id_name]== utt_id ][emotion_name].values.astype("str")[0]) ]
        count_eda_array[current_emotion_id, current_da_id] = count_eda_array[current_emotion_id, current_da_id] + 1    
    print("完成统计后的结果",count_eda_array) 

    pro_eda_array=np.zeros((len(emotion_label),len(da_label)))

    da_results = get_row_statistics(data=data,row_name= da_name)
    emotion_results = get_row_statistics(data=data,row_name= emotion_name)


    for da in da_id:
        for emotion in emotion_id:
            current_da_id = da_id[da]
            current_emotion_id = emotion_id[emotion]
            current_da_number = da_results["element_stastics"][da]
            #current_emotion_number = emotion_results["element_stastics"][emotion]
            pro_eda_array[current_emotion_id, current_da_id] = count_eda_array[current_emotion_id, current_da_id]/(current_da_number)       
    return da_id, emotion_id, count_eda_array, pro_eda_array

