# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-05-30 14:46:36
'''
import copy
import pickle as cPickle

from utils import THRESHOLD


class DataBatch():
    def __init__(self, max_length=100, batch_size=20, data_type='train'):
        self.index = 0
        self.input_size = 0
        self.batch_size = batch_size
        self.max_length = max_length
        self.data_type = data_type
        self.data = []
        self.batch_data = []
        self.vocab = {"unk": 0}
        # self.tag_map = {"O":0, "B-ORG":1, "I-ORG":2, "E-ORG":3, "B-PER":4, "I-PER":5, "E-PER":6, "S":7}
        self.tag_map = {"O":0}

        if data_type == "train":
            self.data_path = "data/train"
        elif data_type == "dev":
            self.data_path = "data/dev"
            self.load_data_map()
        elif data_type == "test":
            self.data_path = "data/test"
            self.load_data_map()

        self.load_data()
        self.prepare_batch()

    def load_data_map(self):
        with open("data/data_map.pkl", "rb") as f:
            self.data_map = cPickle.load(f)
            self.vocab = self.data_map.get("vocab", {})
            self.tag_map = self.data_map.get("tag_map", {})

    def load_data(self):
        # load data
        # add vocab
        # covert to one-hot
        sentence = []
        target = []
        train_nums = 0
        with open(self.data_path) as f:
            line = f.readline()
            while line:
                train_nums += 1
                line = line.rstrip()
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
                    line = f.readline()
                    continue
                # 添加字典
                if word not in self.vocab and self.data_type == "train":
                    self.vocab[word] = max(self.vocab.values()) + 1 
                sentence.append(self.vocab.get(word, 0)) 
                target.append(tag)
                line = f.readline()
        self.input_size = len(self.vocab.values())
        print("-"*50)
        print("{} data: {}".format(self.data_type ,len(self.data)))
        print("vocab size: {}".format(self.input_size))
        print("unique tag: {}".format(len(self.tag_map.values())))
    
    def convert_tag(self, data):
        # add E-XXX for tags
        # add O-XXX for tags
        _, tags = data
        converted_tags = []
        for _, tag in enumerate(tags[:-1]):
            if tag not in self.tag_map and self.data_type == "train":
                self.tag_map[tag] = len(self.tag_map.keys())
            converted_tags.append(self.tag_map.get(tag, 0))
        converted_tags.append(0)
        data[1] = converted_tags
        assert len(converted_tags) == len(tags), "convert error, the list dosen't match!"
        return data

    def prepare_batch(self):
        '''
            prepare data for batch
        '''
        index = 0
        while True:
            if index+self.batch_size >= len(self.data):
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index:index+self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)
    
    def pad_data(self, data):
        c_data = copy.deepcopy(data)

        max_length = max([len(i[0]) for i in c_data])
        for i in c_data:
            i[0] = i[0] + (max_length-len(i[0])) * [0]
            i[1] = i[1] + (max_length-len(i[1])) * [0]
        return c_data

    def iteration(self):
        idx = 0
        while True:
            yield self.batch_data[idx]
            idx += 1
            if idx > len(self.batch_data)-1:
                idx = 0

    def get_batch(self):
        for data in self.batch_data:
            yield data
