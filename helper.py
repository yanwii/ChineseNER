# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2019-02-26 10:48:20
'''

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--bert", type=bool, help="google bert pretrained model")
parser.add_argument("-dd", "--data_dir", type=str, help="tran, dev and test data dir")
parser.add_argument("-bc", "--bert_config", type=str, help="bert config file dir")
parser.add_argument("-od", "--output_dir", type=str, help="output dir")
parser.add_argument("-ic", "--init_checkpoint", type=str, help="bert model dir")
parser.add_argument("-v", "--vocab_dir", type=str, help="vocab dir")
ARGS = parser.parse_args()