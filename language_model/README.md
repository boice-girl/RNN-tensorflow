# 基于循环神经网络的神经语言模型
- `simple-example`文件夹中是PTB（Peen Treebank Dataset）文本数据集，我们主要使用`data`子文件夹中的三个文件：`ptb.train.txt`、`ptb.valid.txt`、`ptb.test.txt`。
- `count.py`用于统计单词的频率，并且按照词频顺序将词汇表保存到独立的vocab文件中。
- `assign.py`用于将单词转换为编号（即在词汇文件中的行号），并且写入新的文件中。
- `train.py`是训练程序，使用一个双层的LSTM作为循环神经网络的主体，并且共享Softmax层和词向量层的参数， 使用`log perplexity`作为优化的目标函数。
