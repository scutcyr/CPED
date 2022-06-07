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
# File: cped_util.py
# Used for CPED dataset loading
# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2022.03.22

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

# 关键包版本说明：
# pytorch: 1.9.0+

import os
import re
import torch
import logging
import pandas as pd

logger = logging.getLogger(__name__)


# CPED数据采集用到的一些常量，例如：情感标签、对话动作、
CPED_SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]
CPED_IGNORE_ID = -1 # Tokens with indices set to ``-1`` are ignored，用于训练SpeakerBert

# 以下列表用于添加到词表当中
CPED_DA_TOKENS = ["[greeting]","[question]","[answer]","[statement-opinion]","[statement-non-opinion]","[apology]",
                  "[command]","[agreement]","[disagreement]","[acknowledge]","[appreciation]","[interjection]",
                  "[conventional-closing]","[quotation]","[reject]","[irony]","[comfort]","[thanking]","[da-other]"]  # 19 DA labels

CPED_SENTIMENT_TOKENS = ["[neutral]","[positive]","[negative]"]

CPED_EMOTION_TOKENS = ["[happy]","[grateful]","[relaxed]","[positive-other]","[anger]","[sadness]","[fear]",
                       "[depress]","[disgust]","[astonished]","[worried]","[negative-other]","[neutral]"] # 13 emotion labels

CPED_DA_TO_TOKENS = {'greeting': '[greeting]', 'question': '[question]', 'answer': '[answer]', 
                     'statement-opinion': '[statement-opinion]', 'statement-non-opinion': '[statement-non-opinion]', 
                     'apology': '[apology]', 'command': '[command]', 'agreement': '[agreement]', 
                     'disagreement': '[disagreement]', 'acknowledge': '[acknowledge]', 'appreciation': '[appreciation]', 
                     'interjection': '[interjection]', 'conventional-closing': '[conventional-closing]', 
                     'quotation': '[quotation]', 'reject': '[reject]', 'irony': '[irony]', 
                     'comfort': '[comfort]','thanking':'[thanking]', 'other': '[da-other]'}

CPED_SENTIMENT_TO_TOKENS = {'neutral': '[neutral]', 'positive': '[positive]', 'negative': '[negative]'}

CPED_EMOTION_TO_TOKENS = {'happy': '[happy]', 'grateful': '[grateful]', 'relaxed': '[relaxed]', 
                          'positive-other': '[positive-other]', 'anger': '[anger]', 'sadness': '[sadness]', 
                          'fear': '[fear]', 'depress': '[depress]', 'disgust': '[disgust]', 
                          'astonished': '[astonished]', 'worried': '[worried]', 'negative-other': '[negative-other]', 
                          'neutral': '[neutral]'}

CPED_DA_TO_ID = {'greeting': 0, 'question': 1, 'answer': 2, 'statement-opinion': 3, 'statement-non-opinion': 4, 
            'apology': 5, 'command': 6, 'agreement': 7, 'disagreement': 8, 'acknowledge': 9, 'appreciation': 10, 
            'interjection': 11, 'conventional-closing': 12, 'quotation': 13, 'reject': 14, 'irony': 15, 
            'comfort': 16,'thanking':17, 'other': 18}

CPED_EMOTION_TO_ID = {'happy': 0, 'grateful': 1, 'relaxed': 2, 'positive-other': 3, 'anger': 4, 'sadness': 5, 
                 'fear': 6, 'depress': 7, 'disgust': 8, 'astonished': 9, 'worried': 10, 
                 'negative-other': 11, 'neutral': 12}

CPED_GENDER_TO_ID = {'female': 0, 'unknown': 1, 'male': 2}
CPED_BIGFIVE_TO_ID = {'low': 0, 'unknown': 1, 'high': 2}

CPED_SPEAKER_TYPE_TO_ID ={"[speaker1]": 0, "[speaker2]": 1, "[MASK]": 2}

# 给语音识别文本加上标点符号
# https://blog.csdn.net/qq_33200967/article/details/122474859





def tokenize(utterance, tokenizer):
    '''tokenize：使用tokenizer对utterance进行tokenize
    Inputs:
        utterance: 字符串，一个句子
        tokenizer: Tokenizer对象，参考：https://huggingface.co/docs/transformers/main_classes/tokenizer
    Outputs:
        ids: 列表，列表中的每个元素对应token的id，例如：[2108, 3342, 3342, 8024, 1962, 1008, 2769, 1420, 1127, 1127, 6432, 6814]
    Example:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        ids = tokenize(utterance="季杨杨，好像我听凡凡说过", tokenizer=tokenizer)
        print(ids)
        # 返回：[2108, 3342, 3342, 8024, 1962, 1008, 2769, 1420, 1127, 1127, 6432, 6814]
    '''
    utterance = str(utterance)  # 保证为str类型
    # 对于问句添加问号
    utterance = utterance.replace("吗", "吗？")
    utterance = utterance.replace("？？", "？")

    # 对于感叹句添加感叹号
    utterance = utterance.replace("啊", "啊！")
    utterance = utterance.replace("吧", "吧！")
    utterance = utterance.replace("啦", "啦！")
    utterance = utterance.replace("呀", "呀！")
    utterance = utterance.replace("！！", "！")

    # 对于句子中间非问句，非感叹句添加逗号
    utterance = utterance.replace(" ", "，")
    # 去除重复标点符号
    utterance = utterance.split()  # 去除全部空格

    utt_list = list(utterance)  # "季杨杨，好像我听凡凡说过" --> ['季', '杨', '杨', '，', '好', '像', '我', '听', '凡', '凡', '说', '过']

    utterance = ' '.join(utt_list)  # ['季', '杨', '杨', '，', '好', '像', '我', '听', '凡', '凡', '说', '过']--> “季 杨 杨 ， 好 像 我 听 凡 凡 说 过”  # <class 'str'>
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utterance))


def cped_get_single_file(file_path,
                         tokenizer,
                         logger,
                         usecols=["Dialogue_ID","Utterance_ID","Speaker","Sentiment","Emotion","DA","Utterance","Gender","Age","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                         args=None):
    '''cped_get_single_file: 读取指定路径的csv文件，例如：CPED_total_text.csv、train_split.csv、...
    Inputs:
        file_path: 字符串，指定文件路径
        tokenizer: Tokenizer对象，参考：https://huggingface.co/docs/transformers/main_classes/tokenizer
        logger:  logging日志对象
        usecols: 列表，列表中的字符串指定了读取的csv文件的列名，其中
                 "Dialogue_ID","Utterance_ID","Speaker","Utterance"
                 是必需项
        args: parser.parse_args()返回的参数字典
    Outputs:
        data: DataFrame对象
        samples: DataFrame对象
    Example:
        import logging
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        logger = logging.getLogger(__name__)
        file_path = "../data/CPED/test_split.csv"
        data, samples = cped_get_single_file(file_path, tokenizer, logger)

    '''
    logger.info("Read file from %s", file_path)
    data = pd.read_csv(file_path, 
                       usecols=usecols, 
                       encoding="UTF-8-SIG")
    samples = data.iloc[0:30]

    logger.info("Start tokenizing and encoding the file")
    data["Token"] = [tokenize(s, tokenizer) for s in data["Utterance"]]
    logger.info("Finished tokenizing and encoding the dataset")
    return data, samples


def cped_get_single_cache_file(file_path,
                               cache_path,
                               tokenizer,
                               logger,
                               usecols=["Dialogue_ID","Utterance_ID","Speaker","Sentiment","Emotion","DA","Utterance","Gender","Age","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                               args=None):
    '''cped_get_single_cache_file: 读取指定路径的csv文件,如果存在cache，则直接读取cache文件，
                                   例如：CPED_total_text.csv、train_split.csv、...
       这个函数与cped_get_single_file的最大不同就是，第一次读取会保存一个cache文件，之后再读取，就不需要
       调用tokenizer进行预处理了，节省大量实验时间
    Inputs:
        file_path: 字符串，指定文件路径
        cache_path: cache文件保存的路径，建议这个文件的命名做好管理，否则容易混淆数据集，torch.save(data, cache_path)
        tokenizer: Tokenizer对象，参考：https://huggingface.co/docs/transformers/main_classes/tokenizer
        logger:  logging日志对象
        usecols: 列表，列表中的字符串指定了读取的csv文件的列名，其中
                 "Dialogue_ID","Utterance_ID","Speaker","Utterance"
                 是必需项
        args: parser.parse_args()返回的参数字典
    Outputs:
        data: DataFrame对象
        samples: DataFrame对象
    Example:
        import logging
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        logger = logging.getLogger(__name__)
        file_path = "../data/CPED/test_split.csv"
        cache_path = "../data/CPED/test_split_cache"
        data, samples = cped_get_single_cache_file(file_path, cache_path, tokenizer, logger)

    '''
    if cache_path and os.path.isfile(cache_path):
        logger.info("Load tokenized dataset from cache at %s", cache_path)
        data = torch.load(cache_path)
        samples = None
    else: # 从原始文件中读取数据
        logger.info("Read dataset from %s", file_path)
        data, samples = cped_get_single_file(file_path=file_path, 
                                             tokenizer=tokenizer, 
                                             logger=logger, 
                                             usecols=usecols,
                                             args=args)
        logger.info("Finished tokenizing and encoding the dataset")
        logger.info("Save tokenized dataset to cache at %s", cache_path)
        torch.save(data, cache_path)
    return data, samples


def cped_get_data_from_dir(dir_path,
                           cache_path,
                           tokenizer,
                           logger,
                           filenames={"train":"train_shuffle_split.csv",
                                      "valid":"valid_shuffle_split.csv",
                                      "test":"test_shuffle_split.csv"},
                           usecols=["Dialogue_ID","Utterance_ID","Speaker","Sentiment","Emotion","DA","Utterance","Gender","Age","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                           args=None):
    '''cped_get_data_from_dir: 读取dir_path指定目录下，字典filenames指定的数据集
                               如果存在cache，则直接读取cache文件，
    Inputs:
        dir_path: 字符串，指定数据集存放的目录
        cache_path: cache文件保存的路径，建议这个文件的命名做好管理，否则容易混淆数据集，torch.save(data, cache_path)
        tokenizer: Tokenizer对象，参考：https://huggingface.co/docs/transformers/main_classes/tokenizer
        logger:  logging日志对象
        filenames: 字典，包括"train"、"valid"、"test"三个键，其值指定对应的文件名
        usecols: 列表，列表中的字符串指定了读取的csv文件的列名，其中
                 "Dialogue_ID","Utterance_ID","Speaker","Utterance"
                 是必需项
        args: parser.parse_args()返回的参数字典
    Outputs:
        data: 字典，格式为{"train":train_data,"valid":valid_data, "test":test_data}，每一个值为DataFrame对象
        samples: DataFrame对象
    Example:
        import logging
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        logger = logging.getLogger(__name__)
        dir_path = "../data/CPED"
        cache_path = "../data/CPED/cped_cache"
        filenames = {"train":"train_shuffle_split.csv",
                     "valid":"valid_shuffle_split.csv",
                     "test":"test_shuffle_split.csv"}
        data, samples = cped_get_data_from_dir(dir_path, cache_path, tokenizer, logger, filenames)

    '''
    if cache_path and os.path.isfile(cache_path):
        logger.info("Load tokenized dataset from cache at %s", cache_path)
        data = torch.load(cache_path)
        samples = None
    else: # 从原始文件中读取数据
        logger.info("Read dataset from %s", dir_path)
        train_data, samples = cped_get_single_file(os.path.join(dir_path,filenames["train"]), tokenizer, logger, usecols, args)
        valid_data, samples = cped_get_single_file(os.path.join(dir_path,filenames["valid"]), tokenizer, logger, usecols, args)
        test_data, samples = cped_get_single_file(os.path.join(dir_path,filenames["test"]), tokenizer, logger, usecols, args)
        data = {"train":train_data,"valid":valid_data, "test":test_data}
        logger.info("Finished tokenizing and encoding the dataset")
        logger.info("Save tokenized dataset to cache at %s", cache_path)
        torch.save(data, cache_path)
    return data, samples


def cped_get_single_file_for_bert_gpt(file_path,
                                      bert_tokenizer,
                                      gpt_tokenizer,
                                      logger,
                                      usecols=["Dialogue_ID","Utterance_ID","Speaker","Sentiment","Emotion","DA","Utterance","Gender","Age","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                                      args=None):
    '''cped_get_single_file_for_bert_gpt: 读取指定路径的csv文件，例如：CPED_total_text.csv、train_split.csv、...
       并使用两种tokenizer进行tokenize
    Inputs:
        file_path: 字符串，指定文件路径
        bert_tokenizer: Tokenizer对象，参考：https://huggingface.co/docs/transformers/main_classes/tokenizer
        gpt_tokenizer: Tokenizer对象，参考：https://huggingface.co/docs/transformers/main_classes/tokenizer
        logger:  logging日志对象
        usecols: 列表，列表中的字符串指定了读取的csv文件的列名，其中
                 "Dialogue_ID","Utterance_ID","Speaker","Utterance"
                 是必需项
        args: parser.parse_args()返回的参数字典
    Outputs:
        data: DataFrame对象
        samples: DataFrame对象
    Example:
        import logging
        from transformers import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        gpt_tokenizer = BertTokenizer.from_pretrained("openai-gpt")
        logger = logging.getLogger(__name__)
        file_path = "../data/CPED/test_split.csv"
        data, samples = cped_get_single_file_for_bert_gpt(file_path, bert_tokenizer, gpt_tokenizer, logger)

    '''
    logger.info("Read file from %s", file_path)
    data = pd.read_csv(file_path, 
                       usecols=usecols, 
                       encoding="UTF-8-SIG")
    samples = data.iloc[0:30]

    logger.info("Start tokenizing and encoding the file")
    data["Token_bert"] = [tokenize(s, bert_tokenizer) for s in data["Utterance"]]
    data["Token_gpt"] = [tokenize(s, gpt_tokenizer) for s in data["Utterance"]]
    logger.info("Finished tokenizing and encoding the dataset")
    return data, samples


def cped_get_data_from_dir_for_bert_gpt(dir_path,
                                        cache_path,
                                        bert_tokenizer,
                                        gpt_tokenizer,
                                        logger,
                                        filenames={"train":"train_shuffle_split.csv",
                                                   "valid":"valid_shuffle_split.csv",
                                                   "test":"test_shuffle_split.csv"},
                                        usecols=["Dialogue_ID","Utterance_ID","Speaker","Sentiment","Emotion","DA","Utterance","Gender","Age","Neuroticism","Extraversion","Openness","Agreeableness","Conscientiousness"],
                                        args=None):
    '''cped_get_data_from_dir_for_bert_gpt: 读取dir_path指定目录下，字典filenames指定的数据集
                                            如果存在cache，则直接读取cache文件，
    Inputs:
        dir_path: 字符串，指定数据集存放的目录
        cache_path: cache文件保存的路径，建议这个文件的命名做好管理，否则容易混淆数据集，torch.save(data, cache_path)
        bert_tokenizer: Tokenizer对象，参考：https://huggingface.co/docs/transformers/main_classes/tokenizer
        gpt_tokenizer: Tokenizer对象，参考：https://huggingface.co/docs/transformers/main_classes/tokenizer
        logger:  logging日志对象
        filenames: 字典，包括"train"、"valid"、"test"三个键，其值指定对应的文件名
        usecols: 列表，列表中的字符串指定了读取的csv文件的列名，其中
                 "Dialogue_ID","Utterance_ID","Speaker","Utterance"
                 是必需项
        args: parser.parse_args()返回的参数字典
    Outputs:
        data: 字典，格式为{"train":train_data,"valid":valid_data, "test":test_data}，每一个值为DataFrame对象
        samples: DataFrame对象
    Example:
        import logging
        from transformers import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        gpt_tokenizer = BertTokenizer.from_pretrained("openai-gpt")
        logger = logging.getLogger(__name__)
        dir_path = "../data/CPED"
        cache_path = "../data/CPED/cped_cache"
        filenames = {"train":"train_shuffle_split.csv",
                     "valid":"valid_shuffle_split.csv",
                     "test":"test_shuffle_split.csv"}
        data, samples = cped_get_data_from_dir_for_bert_gpt(dir_path, cache_path, bert_tokenizer, gpt_tokenizer, logger, filenames)

    '''
    if cache_path and os.path.isfile(cache_path):
        logger.info("Load tokenized dataset from cache at %s", cache_path)
        data = torch.load(cache_path)
        samples = None
    else: # 从原始文件中读取数据
        logger.info("Read dataset from %s", dir_path)
        train_data, samples = cped_get_single_file_for_bert_gpt(os.path.join(dir_path,filenames["train"]), bert_tokenizer, gpt_tokenizer, logger, usecols, args)
        valid_data, samples = cped_get_single_file_for_bert_gpt(os.path.join(dir_path,filenames["valid"]), bert_tokenizer, gpt_tokenizer, logger, usecols, args)
        test_data, samples = cped_get_single_file_for_bert_gpt(os.path.join(dir_path,filenames["test"]), bert_tokenizer, gpt_tokenizer, logger, usecols, args)
        data = {"train":train_data,"valid":valid_data, "test":test_data}
        logger.info("Finished tokenizing and encoding the dataset")
        logger.info("Save tokenized dataset to cache at %s", cache_path)
        torch.save(data, cache_path)
    return data, samples

