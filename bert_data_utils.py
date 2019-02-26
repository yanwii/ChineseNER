# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2019-02-26 11:44:15
'''
class BertDataUtils(object):
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.data = []
        self.batch_data = []
    
    def load_data(self):
        # load data
        # add vocab
        # covert to one-hot
        sentence = []
        target = []
        train_nums = 0
        with open(self.data_path) as f:
            for line in f:
                line = line.rstrip()
                train_nums += 1
                try:
                    word, tag = line.split()
                except Exception as error:
                    word = "。"
                    tag = "O" 
                    if line == "end":    
                        converted_data = self.convert_tag([sentence, target])
                        self.data.append(converted_data)
                        sentence = []
                        target = []
                    continue
                if word not in self.vocab and self.data_type == "train":
                    self.vocab[word] = max(self.vocab.values()) + 1 
                sentence.append(self.vocab.get(word, 0)) 
                target.append(tag)
        self.input_size = len(self.vocab.values())
        print("-"*50)
        print("{} data: {}".format(self.data_type ,len(self.data)))
        print("vocab size: {}".format(self.input_size))
        print("unique tag: {}".format(len(self.tag_map.values())))
