import os
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import gradio as gr


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


dataset = 'data/THUCnews'
labels_name = [w.strip() for w in open(os.path.join(dataset, 'class.txt'), 'r', encoding='utf8').readlines()]
device = torch.device('mps')  # 设备cpu
bert_path = 'bert-base-chinese'
num_classes = len(labels_name)
save_path = 'saved_dict/bert.ckpt'
pad_size = 32

# 用预训练的BERT模型进行初始化
tokenizer = BertTokenizer.from_pretrained(bert_path)
model = Model(bert_path, 768, num_classes).to(device)

# 加载模型权重
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()


def predict(sentence):
    inputs = tokenizer.encode(text=sentence, max_length=pad_size, truncation=True,
                              truncation_strategy='longest_first',
                              add_special_tokens=True, pad_to_max_length=True)
    data_encode = (inputs, [1 if x != 0 else 0 for x in inputs])
    input_id = torch.LongTensor(data_encode[0]).unsqueeze(0).to(device)
    input_mask = torch.LongTensor(data_encode[1]).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model((input_id, input_mask))
        predicts = torch.max(outputs.data, 1)[1].cpu()

    return labels_name[predicts[0]]


# gradio封装
iface = gr.Interface(fn=predict, inputs="text", outputs="text")
iface.launch()
