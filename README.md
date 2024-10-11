# Chinese-Text-Classification-Pytorch
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch，开箱即用。

先看原项目：https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch.git

本项目增加了预测类 `my_classifier.py`

## 介绍

数据以字为单位输入模型，预训练词向量使用 [搜狗新闻 Word+Character 300d](https://github.com/Embedding/Chinese-Word-Vectors)，[点这里下载](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ)  

在 utils.py 文件中可以提取预训练词向量

## 环境

- python 3.12
- cuda 12.1

`pip install -r requirements.txt` 安装依赖，若安装的 Pytorch 不支持 CUDA，先卸载 `pip uninstall torch`，后安装 `pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html`

## 更换自己的数据集
 - 如果用字，按照我数据集的格式来格式化你的数据。  
 - 如果用词，提前分好词，词之间用空格隔开，`python run.py --model TextCNN --word True`  
 - 使用预训练词向量：utils.py的main函数可以提取词表对应的预训练词向量。

## 使用说明
```E
# 训练并测试：
# TextCNN 89个品目平均82%准确率
python run.py --model TextCNN --embedding random

# TextRNN 89个品目平均83%准确率
python run.py --model TextRNN

# TextRNN_Att 89个品目平均84%准确率
python run.py --model TextRNN_Att --embedding random

# TextRCNN 89个品目平均82%准确率
python run.py --model TextRCNN --embedding random

# FastText 89个品目86准确率
python run.py --model FastText --embedding random

# DPCNN
python run.py --model DPCNN --embedding random

# Transformer
python run.py --model Transformer --embedding random
```

### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。

## 模型使用
`python my_classifier.py`

[更轻量的server项目](https://github.com/AriesYB/keyword_classifier.git)