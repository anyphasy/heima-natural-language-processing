import random

import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader, Dataset
from rich import print
from transformers import AdamW
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else "cpu"
# device = 'cuda' if torch.cuda.is_available() else "mps"
print(device)
# 加载分词器
bert_tokenizer = BertTokenizer.from_pretrained('../dm08_transformers/bert-base-chinese')
# print('vocab', bert_tokenizer.get_vocab())
print('vocab_size', bert_tokenizer.vocab_size)
bert_model = BertModel.from_pretrained('../dm08_transformers/bert-base-chinese')
bert_model = bert_model.to(device)
# print('mask', bert_tokenizer.mask_token)
# print('mask', bert_tokenizer.mask_token_id)

# 自定义dataset
class MyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        train_dataset = load_dataset('csv', data_files=data_path, split='train')
        # 筛选大于44的样本
        self.train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 44)

        self.sample_len = len(self.train_dataset)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, item):
        label = 1
        sequence = self.train_dataset[item]["text"]
        sent1 = sequence[:22]
        sent2 = sequence[22:44]
        # 构造负样本
        if random.randint(0, 1) == 0:
            j = random.randint(0, self.sample_len-1)
            sent2 = self.train_dataset[j]["text"][22:44]
            label=0
        return sent1, sent2, label


def collate_fn3(data):
    sents = [i[:2] for i in data]
    # print(f'sents-->{sents}')
    labels = [i[-1] for i in data]
    # print(labels)
    inputs = bert_tokenizer.batch_encode_plus(sents, padding="max_length", truncation=True,
                                              max_length=50, return_tensors='pt')
    # print(f'inputs--》{inputs}')
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, token_type_ids, labels


def test_dataset():
    my_dataset = MyDataset(data_path='./data/train.csv')
    # print(len(my_dataset))
    # sent1, sent2, label = my_dataset[3]
    # print(f'sent1--》{sent1}')
    # print(f'sent2--》{sent2}')
    # print(f'label--》{label}')
    my_dataloader = DataLoader(my_dataset, batch_size=8, collate_fn=collate_fn3,
                                drop_last=True, shuffle=True)

    for input_ids, attention_mask, token_type_ids, labels in my_dataloader:
        print(f'input_ids-->{input_ids.shape}')
        print(f'attention_mask-->{attention_mask.shape}')
        print(f'token_type_ids-->{token_type_ids.shape}')
        print(f'labels-->{labels}')
        break

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 768代表bert模型的输出结果
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 我们不更新预训练模型的参数
        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(f'bert_output--》{bert_output}')
        # print(f'bert_output.last_hidden_state-->{bert_output.last_hidden_state.shape}')
        # print(f'bert_output.pooler_output-->{bert_output.pooler_output.shape}')
        # 对句子分类，直接取出pooler_output结果（对应的就是CLS这个token的向量表示————>可以代表整个句子的特征）
        # output-->[8, 2]
        output = self.linear(bert_output.pooler_output)
        return output
def test_model():
    my_dataset = MyDataset(data_path='./data/train.csv')
    # print(len(my_dataset))
    # sent1, sent2, label = my_dataset[3]
    # print(f'sent1--》{sent1}')
    # print(f'sent2--》{sent2}')
    # print(f'label--》{label}')
    my_dataloader = DataLoader(my_dataset, batch_size=8, collate_fn=collate_fn3,
                                drop_last=True, shuffle=True)

    my_model = MyModel()
    for input_ids, attention_mask, token_type_ids, labels in my_dataloader:
        output = my_model(input_ids, attention_mask, token_type_ids)
        print(f'output--》{output}')
        break

def model2train():
    # 实例化模型
    my_model = MyModel()
    my_model = my_model.to(device)
    # 实例化优化器
    my_adamw = AdamW(my_model.parameters(), lr=5e-4)
    # 实例化损失函数
    my_crossentropy = nn.CrossEntropyLoss()
    # 实例化dataset数据源(自定义)
    my_dataset = MyDataset(data_path='./data/train.csv')
    # 不更新预训练模型的参数
    for param in bert_model.parameters():
        param.requires_grad_(False)

    # 设置模型为训练模型（因为使用预训练模型）
    my_model.train()

    # 设置训练的轮次
    epochs = 1

    # 开始训练
    for epoch_idx in range(1, epochs+1):
        # 实例化dataloader对象
        my_dataloader = DataLoader(my_dataset, batch_size=8, collate_fn=collate_fn3,
                                   shuffle=True, drop_last=True)
        start_time = time.time()
        # 内部数据迭代
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(my_dataloader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            output = my_model(input_ids, attention_mask, token_type_ids)
            # print(f'output-->{output}')
            # 计算损失
            my_loss = my_crossentropy(output, labels)
            # 梯度清零
            my_adamw.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            my_adamw.step()
            # print(f'labels-->{labels}')
            # 打印日志
            if i % 5 == 0:
                temp = torch.argmax(output, dim=-1)
                acc = (temp == labels).sum().item() / len(labels)
                print('轮次:%d 迭代数:%d 损失:%.6f 准确率%.3f 时间%d' \
                      %(epoch_idx, i, my_loss.item(), acc, (int)(time.time())-start_time))


        # 保存模型
        torch.save(my_model.state_dict(), './save_model/ai20_nsp_%d.bin' % epoch_idx)

def model2test():
    # 加载测试集dataset对象
    test_dataset = MyDataset(data_path='./data/test.csv')
    # 加载训练好的模型
    my_model = MyModel()
    my_model.load_state_dict(torch.load('./save_model/ai20_nsp_1.bin', map_location='cpu'))

    # 实例化test_dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn3,
                                 drop_last=True, shuffle=True)

    # 设定模型为评估模式
    my_model.eval()
    # 设定几个变量
    correct = 0 # 预测正确的样本个数
    total = 0 # 已经迭代样本总个数

    # 开始测试
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(test_dataloader)):
        # input_ids-->[8, 50]
        with torch.no_grad():
            output = my_model(input_ids, attention_mask, token_type_ids)
        # print(f'output-->{output}')
        # 找出预测概率值对应的索引output-->[8, 2]
        temp_idx = torch.argmax(output, dim=-1)
        # print(f'temp_idx--》{temp_idx}')
        correct = correct + (temp_idx == labels).sum().item()
        total = total + len(labels)

        if i % 20 == 0:
            print(correct / total)

if __name__ == '__main__':
    # test_dataset()
    # test_model()
    # model2train()
    model2test()