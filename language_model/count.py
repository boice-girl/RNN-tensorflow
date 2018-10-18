# -*- coding: utf-8 -*-
import codecs
import collections
from operator import itemgetter

RAW_DATA = "./simple-examples/data/ptb.train.txt"
VOCAB_OUTPUT = "ptb.train.vocab"

counter = collections.Counter()        #统计单词出现的频率,制作词汇表
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
sorted_words = ["<eos>"] + sorted_words  # 在文本换行处添加句子结束符,提前加入词汇表
with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')
