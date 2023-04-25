import logging
from tqdm import tqdm
import os
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
import torch
from transformers import AdamW
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import f1_score

# 初始化log
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

if not os.path.exists('./log/'):
    os.mkdir('./log/')
fh = logging.FileHandler('./log/log.log', mode='a', encoding='utf-8')
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
console.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(console)

# 各项参数设置
dataset = 'data/THUCnews'
labels_name = [w.strip() for w in open(os.path.join(dataset, 'class.txt'), 'r', encoding='utf8').readlines()]
pad_size = 32
# device = torch.device('mps')  # 设备mac gpu
# device = torch.device('cpu')  # 设备cpu
device = torch.device('cuda')  # 设备gpu
batch_size = 64
bert_path = 'bert-base-chinese'
num_classes = len(labels_name)
weight_decay = 0.02
num_epochs = 30
learning_rate = 5e-5
bert_lr_ratio = 0.2
dropout = 0.1
patience = 6
save_path = 'saved_dict/bert.ckpt'


# 读取数据集
def data_loader(file_path):
    contents = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            contents.append((content, label))
    return contents


# 数据encode
def encode_data(tokenizer, data):
    data_encoded = []
    for text, label in tqdm(data, total=len(data)):
        inputs = tokenizer.encode(text=text, max_length=pad_size, truncation=True,
                                  truncation_strategy='longest_first',
                                  add_special_tokens=True, pad_to_max_length=True)
        data_encoded.append((inputs, [1 if x != 0 else 0 for x in inputs], int(label)))
    return data_encoded


# 模型网络搭建
class Model(nn.Module):
    def __init__(self, bert_path, hidden_size, num_classes):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, return_dict=False)
        out = self.fc(pooled)
        return out


def get_optimizer(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_param_ids = list(map(id, param_optimizer))
    no_weight_decay_params = [x[1] for x in filter(
        lambda name_w: any(nwd in name_w[0] for nwd in no_decay), model.named_parameters())]
    no_weight_decay_param_ids = list(map(id, [x[1] for x in no_weight_decay_params]))

    bert_base_params = filter(lambda p: id(p) in bert_param_ids and id(p) not in no_weight_decay_param_ids,
                              model.parameters())
    bert_no_weight_decay_params = filter(lambda p: id(p) in bert_param_ids and id(p) in no_weight_decay_param_ids,
                                         model.parameters())
    base_no_weight_decay_params = filter(
        lambda p: id(p) not in bert_param_ids and id(p) in no_weight_decay_param_ids,
        model.parameters())
    base_params = filter(lambda p: id(p) not in bert_param_ids and id(p) not in no_weight_decay_param_ids,
                         model.parameters())
    params = [{"params": bert_base_params, "lr": learning_rate * bert_lr_ratio},
              {"params": bert_no_weight_decay_params, "lr": learning_rate * bert_lr_ratio,
               "weight_decay": 0.0},
              {"params": base_no_weight_decay_params, "lr": learning_rate, "weight_decay": 0.0},
              {"params": base_params, "lr": learning_rate}]

    # 设置AdamW优化器
    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    return optimizer


def evaluate(model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_iter)):
            batch = [x.to(device) for x in batch]
            outputs = model((batch[0], batch[1]))
            # print(labels)
            loss = F.cross_entropy(outputs, batch[-1])
            loss_total += loss
            labels = batch[-1].data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all, average='macro')
    report = metrics.classification_report(labels_all, predict_all, target_names=labels_name, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)

    return acc, f1, loss_total / (len(data_iter) + 1e-10), report, confusion


def test(model, test_iter):
    # test
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_acc, test_f1, test_loss, test_report, test_confusion = evaluate(model, test_iter)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test F1:{2:>6.2%}'
    logger.info(msg.format(test_loss, test_acc, test_f1))
    logger.info("Precision, Recall and F1-Score...")
    logger.info(test_report)
    logger.info("Confusion Matrix...")
    logger.info(test_confusion)


def train(model, train_iter, dev_iter, test_iter, optimizer):
    model.train()
    dev_best_f1 = float('-inf')
    last_improve_epoch = 0
    model.train()
    for epoch in range(num_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        # 记录变量
        train_labels_all = np.array([], dtype=int)
        train_predicts_all = np.array([], dtype=int)
        train_loss_list = []
        t = tqdm(train_iter, leave=False, total=len(train_iter), desc='Training')
        for step, batch in enumerate(t):
            batch = [x.to(device) for x in batch]
            model.train()
            model.zero_grad()
            outputs = model((batch[0], batch[1]))
            loss = F.cross_entropy(outputs, batch[-1])
            train_loss_list.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            # 真实标签和预测标签

            predicts = torch.max(outputs.data, 1)[1].cpu()
            labels_train = batch[-1].cpu().data.numpy()
            train_labels_all = np.append(train_labels_all, labels_train)
            train_predicts_all = np.append(train_predicts_all, predicts)

        # 训练集评估
        train_loss = sum(train_loss_list) / (len(train_loss_list) + 1e-10)
        train_acc = metrics.accuracy_score(train_labels_all, train_predicts_all)
        train_f1 = metrics.f1_score(train_labels_all, train_predicts_all, average='macro')

        dev_acc, dev_f1, dev_loss, report, confusion = evaluate(model, dev_iter)
        msg = 'Train Loss: {0:>5.6},  Train Acc: {1:>6.4%},  Train F1: {2:>6.4%},  Val Loss: {3:>5.4},  Val Acc: {4:>6.4%},  Val F1: {5:>6.4%}'
        logger.info(msg.format(train_loss, train_acc, train_f1, dev_loss, dev_acc, dev_f1))
        logger.info("Precision, Recall and F1-Score...")
        logger.info(report)
        logger.info("Confusion Matrix...")
        logger.info(confusion)

        if dev_f1 > dev_best_f1:
            dev_best_f1 = dev_f1
            torch.save(model.state_dict(), save_path)
            last_improve_epoch = epoch

        if epoch - last_improve_epoch > patience:
            logger.info("No optimization for a long time, auto-stopping...")
            break

    test(model, test_iter)


if __name__ == '__main__':
    train_data = data_loader(os.path.join(dataset, 'train.txt'))
    dev_data = data_loader(os.path.join(dataset, 'dev.txt'))
    test_data = data_loader(os.path.join(dataset, 'test.txt'))

    tokenizer = BertTokenizer.from_pretrained(bert_path)

    train_data = encode_data(tokenizer, train_data)
    dev_data = encode_data(tokenizer, dev_data)
    test_data = encode_data(tokenizer, test_data)

    # 迭代器包装
    train_dataset = TensorDataset(*[torch.LongTensor(x).to(device) for x in zip(*train_data)])
    dev_dataset = TensorDataset(*[torch.LongTensor(x).to(device) for x in zip(*dev_data)])
    test_dataset = TensorDataset(*[torch.LongTensor(x).to(device) for x in zip(*test_data)])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 初始化模型
    model = Model(bert_path, 768, num_classes).to(device)
    optimizer = get_optimizer(model)

    train(model, train_loader, dev_loader, test_loader, optimizer)
