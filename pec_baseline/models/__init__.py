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

# Models
# Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>
# Date: 2022.04.06

__version__ = "1.0.0"

# 关键包版本说明：
# pytorch: 1.9.0+
# transformers: 4.11.3+

from .base_model import (is_torch_available)

# 模型类
if is_torch_available():
	from . import (
		gpt,
		gpt2,
		cvgpt
	)

# Model parameters calculating and freezing
from .model_parameters import (count_trainable_parameters, count_total_parameters, show_trainable_parameters, 
                                set_freeze_by_names, freeze_by_model_name, unfreeze_by_model_name)