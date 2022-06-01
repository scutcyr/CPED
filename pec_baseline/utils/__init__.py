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

# Dataset data loading module
# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2022.03.21

__version__ = "1.0.0"

# 关键包版本说明：
# pytorch: 1.9.0+


try:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    absl.logging._warn_preinit_stderr = False
except:
    pass

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# Files and general utilities
from .base_util import (load_csv_data_from_dir,shuffle_total_data,combine_csv_files,save_speaker,load_speaker,
                        convert_speaker_to_id,convert_id_to_speaker,convert_cache_to_csv,is_torch_available)

from .dataset_statistics import (get_data_for_analysis, get_totaldata_for_analysis, get_row_statistics, cout_dialogue_words)


# Dataset utilities
if is_torch_available():
    # 读取CPED数据集的若干种函数
    from .cped_util import (cped_get_single_file,cped_get_single_cache_file,cped_get_data_from_dir,
                            cped_get_single_file_for_bert_gpt,cped_get_data_from_dir_for_bert_gpt,
                            CPED_SPECIAL_TOKENS,CPED_IGNORE_ID,CPED_DA_TOKENS,CPED_SENTIMENT_TOKENS,
                            CPED_EMOTION_TOKENS,CPED_DA_TO_TOKENS,CPED_SENTIMENT_TO_TOKENS,CPED_EMOTION_TO_TOKENS,
                            CPED_DA_TO_ID,CPED_EMOTION_TO_ID,CPED_GENDER_TO_ID,CPED_BIGFIVE_TO_ID,CPED_SPEAKER_TYPE_TO_ID)

    from .cped_dataset import (CpedDataset, build_cped_dataloaders, find_split_id_of_response, create_speaker_type, 
                               convert_emotion_to_tokens, convert_da_to_tokens, set_da_in_speaker, set_emotion_in_speaker)




