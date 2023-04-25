# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer
import os
from utils.config import config


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = 'data/' + dataset + '/train.txt'  # 训练集
        self.dev_path = 'data/' + dataset + '/dev.txt'  # 验证集
        self.test_path = 'data/' + dataset + '/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            'data/' + dataset + '/class.txt','r',encoding='utf8').readlines()]  # 类别名单
        if not os.path.exists('saved_dict/'): os.mkdir('saved_dict/')
        self.save_path = './saved_dict/' + dataset + '-' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda:' + config["gpu"] if torch.cuda.is_available() else 'cpu')  # 设备

        self.patience = config['patience']
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = config['epochs']  # epoch数
        self.batch_size = config['batch_size']  # mini-batch大小
        self.pad_size = config['pad_size']  # 每句话处理成的长度(短填长切)
        self.learning_rate = config['learning_rate']  # 学习率
        self.bert_lr_ratio = config['bert_lr_ratio']
        self.weight_decay = config['weight_decay']
        self.bert_path = config['pretrained']
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.save_result = config["save_result"]


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask,return_dict=False)
        out = self.fc(pooled)
        return out
