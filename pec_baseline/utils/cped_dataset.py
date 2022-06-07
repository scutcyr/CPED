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

# CPED Dataset data loading file
# File: cped_dataset.py
# Used for CPED dataset loading
# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2022.03.29

# CPED数据集说明
# 数据集存储形式如下：
# CPED/                          顶层文件夹，
#      CPED_total_text.csv       全部对话数据汇总的csv格式文件，
#      speakers.txt              所有说话人姓名汇总集合的文本文件，
#      train_split.csv           训练集，与valid_split.csv、test_split.csv无重叠说话人
#      valid_split.csv           验证集，与train_split.csv、test_split.csv无重叠说话人
#      test_split.csv            测试集，与train_split.csv、valid_split.csv无重叠说话人
#      train_shuffle_split.csv   将CPED_total_text.csv随机打乱以8:1:1切割得到
#      valid_shuffle_split.csv   将CPED_total_text.csv随机打乱以8:1:1切割得到
#      test_shuffle_split.csv    将CPED_total_text.csv随机打乱以8:1:1切割得到
#  上述两种数据集划分方式可以用于不同的研究场景！

# 本文件约定函数命名风格：cped_function_name，例如：
# cped_get_total_data、cped_get_single_file

# 本文件约定类的命名风格：CpedClassName，例如：
# CpedDataset、CpedBertDataset、CpedBertGptDataset

# 关键包版本说明：
# pytorch: 1.9.0+
import torch
import logging
from itertools import chain # 将二维列表转换为一维列表，[[1127, 1127, 6432, 6814],[118, 117, 116]]---> [1127, 1127, 6432, 6814, 118, 117, 116]
from torch.utils.data import Dataset # 参考：https://pytorch.org/docs/1.9.0/data.html#torch.utils.data.Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .cped_util import (tokenize, cped_get_single_file, cped_get_single_cache_file, cped_get_data_from_dir, 
                        cped_get_single_file_for_bert_gpt, cped_get_data_from_dir_for_bert_gpt,
                        CPED_SPECIAL_TOKENS, CPED_IGNORE_ID, CPED_DA_TOKENS, CPED_SENTIMENT_TOKENS,
                        CPED_EMOTION_TOKENS, CPED_DA_TO_TOKENS, CPED_SENTIMENT_TO_TOKENS, CPED_EMOTION_TO_TOKENS,
                        CPED_DA_TO_ID, CPED_EMOTION_TO_ID, CPED_GENDER_TO_ID, CPED_BIGFIVE_TO_ID, CPED_SPEAKER_TYPE_TO_ID)


logger = logging.getLogger(__name__)


def find_split_id_of_response(speaker_list,responder):
    '''find_split_id_of_response
    Inputs:
        speaker_list: 姓名组成的列表，例如：['诺澜', '诺澜', '胡一菲', '胡一菲', '胡一菲', '诺澜', '诺澜']
        responder: 字符串，表示回复者的姓名，例如：'诺澜'
    Outputs:
        split_id: 负整数，范围为-1至-len(speaker_list)+1
    Examples:
        speaker_list = ['诺澜', '诺澜', '胡一菲', '胡一菲', '胡一菲', '诺澜', '诺澜']
        responder = '诺澜'
        split_id= find_split_id_of_response(speaker_list,responder)
        # 返回结果：-2
        # utterance_history = data_index["Token"].tolist()[-max_history_utterances:split_id]
        # reponse = data_index["Token"].tolist()[-split_id:]
    '''
    split_id = -1
    for i in range(-2,-len(speaker_list),-1):
        if speaker_list[i] != responder:
            return split_id
        else:
            split_id = split_id-1
    return -1 # 极端情形，只有一个人说话，则只认为最后一句为response内容

def create_speaker_type(speaker_list,responder=None):
    '''create_speaker_type: 将姓名列表转换为"[speaker1]"、"[speaker2]"组成的列表
    Inputs:
        speaker_list: 姓名组成的列表，例如：['诺澜', '诺澜', '胡一菲', '胡一菲', '胡一菲']
        responder: 字符串，表示回复者的姓名，例如：'诺澜'
    Outputs:
        speaker_type_list: "[speaker1]"、"[speaker2]"组成的列表
    '''
    if responder==None: # 不指定responder
        speaker2 = speaker_list[-1] # 最后一个句子的说话人被定义为回复者
    else:
        speaker2 = responder
    speaker_type_list = []
    for speaker in speaker_list:
        if speaker==speaker2:
            speaker_type_list.append("[speaker2]")  #  "[speaker2]"代表回复者
        else:
            speaker_type_list.append("[speaker1]")
    return speaker_type_list


def convert_emotion_to_tokens(emotion_list, 
                              emotion_type="Emotion", 
                              SELECTED_EMOTION_TO_TOKENS={"Emotion":CPED_EMOTION_TO_TOKENS,
                                                          "Sentiment":CPED_SENTIMENT_TO_TOKENS}):
    '''convert_emotion_to_tokens: 将情感列表转换为词表当中的情感字符
    Inputs:
        emotion_list: 对话的情感标签列，每一个元素表示对应的句子的原始情感标签，例如：["happy","happy"]
        emotion_type: 字符串，表示情感类型，"Emotion"或者"Sentiment"，指定了使用SELECTED_EMOTION_TO_TOKENS
                      当中的某一种字典用于将原始标签转换为TOKENS标签
        SELECTED_EMOTION_TO_TOKENS: 字典，其键为情感类型，值为相应的情感标签转换为Tokens的字典，由.cped_util定义
    Outputs:
        emotion_tokens_list: 经过转换后的情感tokens列表
    '''
    # emotion_tokens_list = [SELECTED_EMOTION_TO_TOKENS[emotion_type][emo] for emo in emotion_list]
    emotion_tokens_list = []
    for emo in emotion_list:
        if emo not in SELECTED_EMOTION_TO_TOKENS[emotion_type]:
            emotion_tokens_list.append("[neutral]")
        else:
            emotion_tokens_list.append(SELECTED_EMOTION_TO_TOKENS[emotion_type][emo])
    return emotion_tokens_list

def convert_da_to_tokens(da_list,
                         da_type="DA",
                         SELECTED_DA_TO_TOKENS={"DA":CPED_DA_TO_TOKENS}):
    '''convert_da_to_tokens: 将DA列表转换为词表当中的DA字符
    Inputs:
        da_list: 对话的DA标签列，每一个元素表示对应的句子的原始DA标签
        da_type: 字符串，表示DA类型，"DA"或者自定义的DA列名称，指定了使用SELECTED_da_TO_TOKENS
                      当中的某一种字典用于将原始标签转换为TOKENS标签
        SELECTED_DA_TO_TOKENS: 字典，其键为DA类型，值为相应的DA标签转换为Tokens的字典，由.cped_util定义
    Outputs:
        da_tokens_list: 经过转换后的DA的tokens列表
    '''
    da_tokens_list = [SELECTED_DA_TO_TOKENS[da_type][da] for da in da_list]
    return da_tokens_list


def set_da_in_speaker(da_ids,input_ids,bos, eos, speaker1, speaker2, pad):
    '''set_da_in_speaker: 仅在说话人标志位叠加DA Embedding
    
    '''
    special_token_ids_list = [bos, eos, speaker1, speaker2]
    new_da_ids = []
    for i,da in enumerate(da_ids):
        if input_ids[i] in special_token_ids_list:
            new_da_ids.append(da_ids[i])
        else:
            new_da_ids.append(pad)
    return new_da_ids

def set_emotion_in_speaker(emotion_ids,input_ids,bos, eos, speaker1, speaker2, pad):
    '''set_emotion_in_speaker: 仅在说话人标志位叠加情感标签 Embedding
    
    '''
    special_token_ids_list = [bos, eos, speaker1, speaker2]
    new_emotion_ids = []
    for i,emotion in enumerate(emotion_ids):
        if input_ids[i] in special_token_ids_list:
            new_emotion_ids.append(emotion_ids[i])
        else:
            new_emotion_ids.append(pad)
    return new_emotion_ids


class CpedDataset(Dataset):
    '''CpedDataset：用于常规对话生成模型，例如CDialGPT,增加情感或者个性或者DA控制，或者增加Embedding进行叠加
                    不适用于SPEAKERBERT、BERTGPT等模型

    Inputs:
        data: DataFrame，经过预处理后的对话数据，每一行为一个句子，data至少包含'Dialogue_ID'、"Token"两列
              其中，data['Dialogue_ID']表示样本编号
              其中data["Token"]为对应的句子经过tokenizer映射的ids，看起来像下面这样，
              [2108, 3342, 3342, 8024, 1962, 1008, 2769, 1420, 1127, 1127, 6432, 6814]
        tokenizer: Tokenizer对象，参考：https://huggingface.co/docs/transformers/main_classes/tokenizer
        emotion_type: 字符串，指定情感列名，有"Sentiment"、"Emotion两种选择"，可自行组建新的情感列
        da_type: 字符串，指定DA列名，目前只有"DA"一种，可自行组建新的DA列
        persona_type: 列表，指定个性列名
        max_history: 最大对话轮数，句子数=2*max_history
        batch_first: 布尔类型，指定batch是否在第一维，也就是(batch,...)
        lm_labels: 布尔类型，指定是否返回response
        with_current_speaker: 布尔类型，指定回复时的说话人
        with_current_persona: 布尔类型，指定回复时的个性
        with_current_emotion: 布尔类型，指定回复时的情感
        with_current_da: 布尔类型，指定回复时的对话动作DA
        with_emotion: 布尔类型，指定情感嵌入
        with_da=False： 布尔类型，指定DA嵌入
        use_speaker_name_as_speaker_list: 布尔类型，指定说话人姓名作为speaker_list
        set_eda_in_speaker： 布尔类型，指定在说话人位置嵌入情感或DA

    Outputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``.
            Indices of input sequence tokens in the vocabulary.
        **token_type_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``.
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices)
        **emotion_ids**: (`optional`, returned when ``with_emotion=True``)
            ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``.
        **da_ids**: (`optional`, returned when ``with_da=True``)
            ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``.
        **current_speaker_id**: (`optional`, returned when ``with_current_speaker=True``)
            ``torch.LongTensor`` of shape ``(batch_size, 1)``.
        **current_persona_ids**: (`optional`, returned when ``with_current_persona=True``)
            ``torch.LongTensor`` of shape ``(batch_size, persona_size)``.
        **current_emotion_id**: (`optional`, returned when ``with_current_emotion=True``)
            ``torch.LongTensor`` of shape ``(batch_size, 1)``.
        **current_da_id**: (`optional`, returned when ``with_current_da=True``)
            ``torch.LongTensor`` of shape ``(batch_size, 1)``.
        **lm_labels**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
    
    Examples:


    '''
    def __init__(self, 
                 data, 
                 tokenizer,
                 emotion_type="Emotion", 
                 da_type="DA",
                 persona_type=["Gender","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness","Age"],
                 max_history=25, # 句子数则为50
                 batch_first=True, 
                 lm_labels=True, 
                 with_current_speaker=False,
                 with_current_persona=False,
                 with_current_emotion=False,
                 with_current_da=False,
                 with_emotion=False, 
                 with_da=False,
                 use_speaker_name_as_speaker_list=False,
                 set_eda_in_speaker=False,
                 set_current_speaker_mask=False,
                 max_word_length=512): # 增加限制总的字符数
        self.data = data
        self.tokenizer = tokenizer
        self.emotion_type = emotion_type # 'Emotion' 情感标签列名
        self.da_type = da_type           # 'DA'      DA标签列名
        self.persona_type = persona_type
        self.with_current_speaker = with_current_speaker
        self.with_current_persona = with_current_persona
        self.with_current_emotion = with_current_emotion
        self.with_current_da = with_current_da
        self.with_emotion=with_emotion   # Whether use emotion to help generate dialogue
        self.with_da=with_da             # Whether use DA to help generate dialogue
        self.max_history = max_history   # Maximum number of dialogue turns
        self.max_history_utterances = 2*max_history # Maximum number of dialogue sentences
        self.max_word_length = max_word_length
        self.use_speaker_name_as_speaker_list = use_speaker_name_as_speaker_list
        self.set_eda_in_speaker = set_eda_in_speaker
        self.set_current_speaker_mask = set_current_speaker_mask
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels
        self.keys = list(set(self.data['Dialogue_ID']))
        self.len = len(self.keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        dialogue_id = self.keys[index] # 当前对话样本编号
        data_index = self.data[self.data['Dialogue_ID']==dialogue_id]
        
        if len(data_index["Speaker"].tolist()) > self.max_history_utterances: # 实际句子数大于self.max_history_utterances
            max_history_utterances = self.max_history_utterances
        else: # 实际句子数小于self.max_history_utterances
            max_history_utterances = len(data_index["Speaker"].tolist())
        # 判断data_index的“句子数+data_index["Token"]的Token数+2”是否大于self.max_word_length
        while len(data_index["Speaker"].tolist()[-max_history_utterances:])+len(list(chain(*data_index["Token"].tolist()[-max_history_utterances:])))+2>self.max_word_length:
            max_history_utterances = max_history_utterances - 1

        speaker_name_list = data_index["Speaker"].tolist()[-max_history_utterances:] # 说话人姓名列表
        responder = speaker_name_list[-1] # 回复者姓名
        responder_token = self.tokenizer.convert_tokens_to_ids(responder) # 整数
        
        # 找出回复内容与历史对话的分割id
        response_split_id = find_split_id_of_response(speaker_name_list,responder) 
        # 历史对话内容，长这样：[[2108, 3342, 3342, 8024], [1962, 1008, 2769, 1420]]
        history_utterance_tokens = data_index["Token"].tolist()[-max_history_utterances:response_split_id]
        # 回复内容，长这样：[[1127, 1127, 6432, 6814],[118, 117, 116]]---> [1127, 1127, 6432, 6814, 118, 117, 116]
        if self.lm_labels:
            response_utterance_tokens = data_index["Token"].tolist()[response_split_id:]
            response_utterance_tokens = list(chain(*response_utterance_tokens)) # 二维列表转一维列表
        else:
            response_utterance_tokens = []
        
        # 创建历史对话对应的history_speaker_types
        if self.use_speaker_name_as_speaker_list: 
            # 使用说话人姓名嵌入，需要把说话人姓名加进词表！
            history_speaker_types = speaker_name_list[-max_history_utterances:response_split_id]
        else:
            # 使用函数create_speaker_type创建姓名嵌入
            # "[speaker2]"表示回复者，"[speaker1]"表示另一个说话人
            history_speaker_types = create_speaker_type(speaker_list=speaker_name_list[-max_history_utterances:response_split_id], responder=responder)
        # 将字符串表示转换为id表示
        history_speaker_tokens = self.tokenizer.convert_tokens_to_ids(history_speaker_types)
        
        # 创建历史对话对应的history_emotion_tokens
        if self.with_emotion: # 需要把情感标签加进词表！
            history_emotion_tokens = convert_emotion_to_tokens(emotion_list=data_index[self.emotion_type].tolist()[-max_history_utterances:response_split_id], 
                                                               emotion_type=self.emotion_type)
            history_emotion_tokens = self.tokenizer.convert_tokens_to_ids(history_emotion_tokens)
        else:
            history_emotion_tokens = []
        
        # 创建历史对话对应的history_da_tokens
        if self.with_da: # 需要把DA标签加进词表！
            history_da_tokens = convert_da_to_tokens(da_list=data_index[self.da_type].tolist()[-max_history_utterances:response_split_id], 
                                                   da_type=self.da_type)
            history_da_tokens = self.tokenizer.convert_tokens_to_ids(history_da_tokens)
        else:
            history_da_tokens = []


        # 创建用于指定回复的情感、DA、个性
        # 以下用于情感、DA与词嵌入共同使用一个Embedding
        current_emotion_token = self.tokenizer.convert_tokens_to_ids(data_index[self.emotion_type].tolist()[-1])
        current_da_token = self.tokenizer.convert_tokens_to_ids(data_index[self.da_type].tolist()[-1])
        # 以下用于情感、DA不与词嵌入共同使用一个Embedding
        current_emotion_id = CPED_EMOTION_TO_ID[data_index[self.emotion_type].tolist()[-1]]
        current_da_id = CPED_DA_TO_ID[data_index[self.da_type].tolist()[-1]]
        if self.with_current_persona:
            current_gender_id = CPED_GENDER_TO_ID[data_index[self.persona_type[0]].tolist()[-1]]
            current_Neuroticism_id = CPED_BIGFIVE_TO_ID[data_index[self.persona_type[1]].tolist()[-1]]
            current_Extraversion_id = CPED_BIGFIVE_TO_ID[data_index[self.persona_type[2]].tolist()[-1]]
            current_Openness_id = CPED_BIGFIVE_TO_ID[data_index[self.persona_type[3]].tolist()[-1]]
            current_Agreeableness_id = CPED_BIGFIVE_TO_ID[data_index[self.persona_type[4]].tolist()[-1]]
            current_Conscientiousness_id = CPED_BIGFIVE_TO_ID[data_index[self.persona_type[5]].tolist()[-1]]
            current_persona_ids = [current_gender_id,current_Neuroticism_id,current_Extraversion_id,current_Openness_id,
                                   current_Agreeableness_id,current_Conscientiousness_id]
        else:
            current_persona_ids = []

        return self.process(history_speaker_tokens, 
                            history_utterance_tokens,
                            history_emotion_tokens,
                            history_da_tokens,
                            responder_token,
                            current_emotion_token,
                            current_da_token,
                            current_emotion_id,
                            current_da_id,
                            current_persona_ids, 
                            response_utterance_tokens)

    def process(self, 
                history_speaker_tokens,
                history_utterance_tokens,
                history_emotion_tokens,
                history_da_tokens,
                responder_token,
                current_emotion_token,
                current_da_token,
                current_emotion_id,
                current_da_id,
                current_persona_ids, 
                response_utterance_tokens,
                with_eos=True):
        instance = {}
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(CPED_SPECIAL_TOKENS)
        speaker_tokens = history_speaker_tokens + [responder_token]
        emotion_tokens = history_emotion_tokens + [current_emotion_token]
        da_tokens = history_da_tokens + [current_da_token]
        sequence = [[bos]] + history_utterance_tokens + [response_utterance_tokens + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker_tokens[i]] + s
                                    for i, s in enumerate(sequence[1:])]  
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker_tokens[i] for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]

        if self.with_da:
            instance["da_ids"] = [bos] + [da_tokens[i] for i, s in
                                          enumerate(sequence[1:])
                                          for _ in s]
            # only set the DA in [speaker1] or [speaker2]
            if self.set_eda_in_speaker:
                instance["da_ids"] = set_da_in_speaker(instance["da_ids"],instance["input_ids"],bos, eos, speaker1, speaker2, self.pad)
        if self.with_emotion:
            instance["emotion_ids"] = [bos] + [emotion_tokens[i] for i, s in
                                               enumerate(sequence[1:])
                                               for _ in s]
            # only set the emotion in [speaker1] or [speaker2]
            if self.set_eda_in_speaker:
                instance["emotion_ids"] = self.set_emotion_in_speaker(instance["emotion_ids"],instance["input_ids"],bos, eos, speaker1, speaker2, self.pad)
        if self.with_current_speaker:
            instance["current_speaker_id"] = responder_token
        
        if self.with_current_emotion:
            instance["current_emotion_id"] = current_emotion_id
        
        if self.with_current_da:
            instance["current_da_id"] = current_da_id

        if self.with_current_persona:
            instance["current_persona_ids"] = current_persona_ids
        
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        if self.set_current_speaker_mask:
            instance["current_speaker_mask"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [1] + ([-1] * len(sequence[-1][1:]) )

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        
        if self.with_emotion:
            emotion_ids = pad_sequence(
                [torch.tensor(instance["emotion_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=self.pad)
        else:
            emotion_ids = None

        if self.with_da:
            da_ids = pad_sequence(
                [torch.tensor(instance["da_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=self.pad)
        else:
            da_ids = None

        if self.with_current_speaker:
            current_speaker_id = torch.tensor(
                [torch.tensor(instance["current_speaker_id"], dtype=torch.long) for instance in batch],
                dtype=torch.long)
        else:
            current_speaker_id = None
        
        if self.with_current_persona:
            current_persona_ids = pad_sequence(
                [torch.tensor(instance["current_persona_ids"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=1) # padding_value=1 means unknown here
        else:
            current_persona_ids = None

        if self.with_current_emotion:
            current_emotion_id = torch.tensor(
                [torch.tensor(instance["current_emotion_id"], dtype=torch.long) for instance in batch],
                dtype=torch.long)
        else:
            current_emotion_id = None
        
        if self.with_current_da:
            current_da_id = torch.tensor(
                [torch.tensor(instance["current_da_id"], dtype=torch.long) for instance in batch],
                dtype=torch.long)
        else:
            current_da_id = None
        lm_labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        
        if self.set_current_speaker_mask: # for CVGPT
            current_speaker_mask = pad_sequence(
                [torch.tensor(instance["current_speaker_mask"], dtype=torch.long) for instance in batch],
                batch_first=self.batch_first, padding_value=-1)
            return input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels, current_speaker_mask
        else:
            return input_ids, token_type_ids, emotion_ids, da_ids, current_speaker_id, current_persona_ids, current_emotion_id, current_da_id, lm_labels


def build_cped_dataloaders(args, 
                           tokenizer, 
                           logger, 
                           load_test=False,
                           filenames={"train":"train_shuffle_split.csv",
                                      "valid":"valid_shuffle_split.csv",
                                      "test":"test_shuffle_split.csv"}):
    data,sample = cped_get_data_from_dir(dir_path=args.data_path,
                                         cache_path=args.cache_path,
                                         tokenizer=tokenizer,
                                         logger=logger,
                                         filenames=filenames)

    if load_test==False:
        logger.info("Build train and validation dataloaders")
        train_data = data["train"]
        valid_data = data["valid"]
        train_dataset = CpedDataset(data=train_data, 
                                    tokenizer=tokenizer, 
                                    emotion_type=args.emotion_type,
                                    da_type=args.da_type, 
                                    persona_type=["Gender","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                                    max_history=args.max_history,
                                    batch_first=True, 
                                    lm_labels=True, 
                                    with_current_speaker=args.with_current_speaker,
                                    with_current_persona=args.with_current_persona,
                                    with_current_emotion=args.with_current_emotion,
                                    with_current_da=args.with_current_da,
                                    with_emotion=args.with_emotion, 
                                    with_da=args.with_da,
                                    use_speaker_name_as_speaker_list=args.use_speaker_name_as_speaker_list,
                                    set_eda_in_speaker=args.set_eda_in_speaker,
                                    set_current_speaker_mask=args.set_current_speaker_mask)


        valid_dataset = CpedDataset(data=valid_data, 
                                    tokenizer=tokenizer, 
                                    emotion_type=args.emotion_type,
                                    da_type=args.da_type, 
                                    persona_type=["Gender","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                                    max_history=args.max_history,
                                    batch_first=True, 
                                    lm_labels=True, 
                                    with_current_speaker=args.with_current_speaker,
                                    with_current_persona=args.with_current_persona,
                                    with_current_emotion=args.with_current_emotion,
                                    with_current_da=args.with_current_da,
                                    with_emotion=args.with_emotion, 
                                    with_da=args.with_da,
                                    use_speaker_name_as_speaker_list=args.use_speaker_name_as_speaker_list,
                                    set_eda_in_speaker=args.set_eda_in_speaker,
                                    set_current_speaker_mask=args.set_current_speaker_mask)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
        train_loader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  collate_fn=train_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.train_batch_size,
                                  shuffle=(not args.distributed))
        valid_loader = DataLoader(valid_dataset, 
                                  sampler=valid_sampler,
                                  collate_fn=valid_dataset.collate,
                                  num_workers=args.num_workers,
                                  batch_size=args.valid_batch_size,
                                  shuffle=False)

        return train_loader, valid_loader, train_sampler, valid_sampler

    else:
        logger.info("Build test dataloaders")
        test_data = data["test"]
        test_dataset = CpedDataset(data=test_data, 
                                   tokenizer=tokenizer, 
                                   emotion_type=args.emotion_type,
                                   da_type=args.da_type, 
                                   persona_type=["Gender","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                                   max_history=args.max_history,
                                   batch_first=True, 
                                   lm_labels=True, 
                                   with_current_speaker=args.with_current_speaker,
                                   with_current_persona=args.with_current_persona,
                                   with_current_emotion=args.with_current_emotion,
                                   with_current_da=args.with_current_da,
                                   with_emotion=args.with_emotion, 
                                   with_da=args.with_da,
                                   use_speaker_name_as_speaker_list=args.use_speaker_name_as_speaker_list,
                                   set_eda_in_speaker=args.set_eda_in_speaker,
                                   set_current_speaker_mask=args.set_current_speaker_mask)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.distributed else None
        test_loader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 collate_fn=test_dataset.collate,
                                 num_workers=args.num_workers,
                                 batch_size=args.test_batch_size,
                                 shuffle=False)
        return test_loader, test_sampler
