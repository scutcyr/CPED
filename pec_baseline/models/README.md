# models
# 模型设计模块
* **Author: Chen Yirong <eeyirongchen@mail.scut.edu.cn>**
* **Date: 2022.03.21**

## 架构说明
每个子文件夹存放一个模型，其中，文件夹命名使用小写字母+下划线+数字的组合，例如：```gpt```、```gpt2```、```gpt_per```。   
每个模型由3个文件组成，假设该模型命名为```xxx```：
* ```__init__.py```: 对外提供可访问的接口
* ```modeling_xxx.py```: 模型类的定义，
* ```tokenization_xxx.py```: 模型相关的tokenization类（不是必须的）

### 基础文件


