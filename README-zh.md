# [CPED](https://github.com/scutcyr/CPED)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python) [![GitHub stars](https://img.shields.io/github/stars/scutcyr/CPED)](https://github.com/scutcyr/CPED/stargazers) [![GitHub license](https://img.shields.io/github/license/scutcyr/CPED)](https://github.com/scutcyr/CPED/blob/main/LICENSE) ![GitHub repo size](https://img.shields.io/github/repo-size/scutcyr/CPED) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![GitHub last commit](https://img.shields.io/github/last-commit/scutcyr/CPED)

README: [English](https://github.com/scutcyr/CPED/blob/main/README.md) | [中文](https://github.com/scutcyr/CPED/blob/main/README-zh.md)
该仓库提供下面的论文的实现细节：    
**[CPED: A Large-Scale Chinese Personalized and Emotional Dialogue Dataset for Conversational AI](https://arxiv.org/abs/2205.14727)**  

更多信息请参考我们的[论文](https://arxiv.org/abs/2205.14727)。

## <a name="#Contents">目录</a>
* <a href="#Introduction">简介</a>
* <a href="#Dataset">数据集统计学特性</a>
* <a href="#Task">任务定义</a>
* <a href="#Evaluation">实验结果</a>
* <a href="#Usage">使用方法</a>

## <a name="#Introduction">简介</a>
我们构建了一个命名为**CPED**的数据集，该数据集源于40部中文电视剧。
CPED包括与情感、个性特质相关的多源知识，包括：13类情绪、性别、大五人格、19类对话动作以及其他知识。下表给出了CPED与其他常见数据集的比较。    

* 我们构建了一个多轮的中文个性情感对话数据集CPED。据我们所知，CPED是首个中文个性情感对话数据集。它包括超过1.2万个对话，超过13.3万个语句，并且是多模态的。因此，该数据集可以用在复杂的对话理解任务以及拟人化的对话生成任务研究。
* CPED提供了3类属性标注（姓名、性别、年龄），大五人格特质标注，2类情感标注（3分类粗粒度情感、13分类细粒度情感），以及对话动作DA标注。人格特质和情感可以用作开放域对话生成的先验外部知识。提升对话系统的拟人化水平。
* 我们在论文中提出了3个任务：对话中的人格识别（PRC），对话中的情感识别（ERC），以及个性情感对话生成（PEC），一系列实验验证了人格以及情感对于对话生成的重要性。

![dataset_comparison](./images/dataset_comparison.png)

（未完待续...）