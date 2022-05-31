# [CPED](https://github.com/scutcyr/CPED)
    
README: [English](https://github.com/scutcyr/CPED/blob/main/README.md)
This repository provides the implementation details for the paper:    
**CPED: A Large-Scale Chinese Personalized and Emotional Dialogue Dataset for Open-domain Conversation**   

For more information, please refer to our [paper](https://arxiv.org/abs/2205.14727)。

## <a name="#Contents">Contents</a>
* <a href="#Introduction">Introduction</a>
* <a href="#Dataset">Dataset Statistics</a>
* <a href="#Model">Baseline Model</a>
* <a href="#Evaluation">Evaluation Results</a>
* <a href="#Usage">Usage</a>

## <a name="#Introduction">Introduction</a>
We construct a dataset named **CPED** from 40 Chinese TV shows. CPED consists of multisource knowledge related to empathy and personal characteristic. This knowledge covers 13 emotions, gender, Big Five personality traits, 19 dialogue acts and other knowledge. The table below shows a comparison of CPED with some other common conversation data sets.

![dataset_comparison](./images/dataset_comparison.png)

## <a name="#Dataset">Dataset Statistics</a>
In order for the dialogue system to learn emotional expression and personalized expression abilities, we provide multiple types of annotation labels listed in the following Table.

| # of annos. | Labels | Num. |
|:-----------:|:-------|:----:|
| Sentiment | positive, neutral, and negative | 3 |
| Emotion | happy, grateful, relaxed, other-positive, neutral, angry, sad, feared, depressed, disgusted, astonished, worried and other-negative | 13 |
| Gender | male, female, and unknown | 3 |
| Age group | children, teenager, young, middle-aged, elderly and unknown | 6 |
| Big Five | high, low, and unknown | 3 |
| DA | greeting (g), question (q), answer (ans), statement-opinion (sv), statement-non-opinion (sd), apology (fa), command (c), agreement/acceptance (aa), disagreement (dag), acknowledge (a), appreciation (ba), interjection (ij), conventional-closing (fc), thanking (ft), quotation (^q), reject(rj), irony (ir), comfort (cf) and other (oth) | 19 |
| Scene | home, office, school, mall, hospital, restaurant, sports-venue, entertainment-venue, car, outdoor and other-scene | 11 |


Distribution of Gender, Age Group, Sentiment, Emotion and DA in CPED Dataset are shown in the following figure.
![](./images/dataset_staticstics.png)

 The statistics of CPED are listed in the folloeing table.
| Statistics                      | Train   | Dev     | Test    |
|---------------------------------|---------|---------|---------|
| # of modalities                 | (v,a,t) | (v,a,t) | (v,a,t) |
| # of TV plays                   | 26      | 5       | 9       |
| # of dialogues                  | 8,086   | 934     | 2,815   |
| # of utterances                 | 94,187  | 11,137  | 27,438  |
| # of speakers                   | 273     | 38      | 81      |
| Avg. # utt. per dial.           | 11.6    | 11.9    | 9.7     |
| Max # utt. per dial.            | 75      | 31      | 34      |
| Avg. # of emot. per dial.       | 2.8     | 3.4     | 3.2     |
| Avg. # of DAs per dial.         | 3.6     | 3.7     | 3.2     |
| Avg. utt. length                | 8.3     | 8.2     | 8.3     |
| Max utt. length                 | 127     | 42      | 45      |
| Avg. duration of an utterance   | 2.1s    | 2.12s   | 2.21s   |




>>> Engineering Research Ceter of Ministry of Education on Human Body Perception
