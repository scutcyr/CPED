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

# Basic model configuration file
# File: base_model.py
# Used for model configuration
# 用于数据集读取的基础方法
# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2022.04.06

import os
import logging
import importlib.util

logger = logging.getLogger(__name__)

_torch_available = importlib.util.find_spec("torch") is not None

def is_torch_available():
    return _torch_available