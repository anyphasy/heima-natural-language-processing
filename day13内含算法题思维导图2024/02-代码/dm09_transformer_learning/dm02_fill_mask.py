import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader
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
# print('mask', bert_tokenizer.mask_token)
# print('mask', bert_tokenizer.mask_token_id)
# 记载模型
bert_model = BertModel.from_pretrained('../dm08_transformers/bert-base-chinese')
bert_model = bert_model.to(device)
# print(f'bert_model--》{bert_model}')

def collate_fn2(data):
    sents = [i["text"] for i in data]
    inputs = bert_tokenizer.batch_encode_plus(sents, truncation=True, padding="max_length",
                                              max_length=32, return_tensors='pt')

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # print(f'attention_mask--》{attention_mask.shape}')
    token_type_ids = inputs["token_type_ids"]
    # print(f'token_type_ids--》{token_type_ids.shape}')
    # 取出每个样本的第16个位置进行完形填空任务的训练
    labels = input_ids[:, 16].view(-1).clone()
    input_ids[:, 16] = bert_tokenizer.get_vocab()[bert_tokenizer.mask_token]
    # print(f'input_ids--》{input_ids}')
    labels = torch.tensor(labels, dtype=torch.long)
    # print(labels)
    return input_ids, attention_mask, token_type_ids, labels


def test_dataset():
    # 加载训练数据集
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # print(f'train_dataset--》{train_dataset[:3]}')
    # print(f'train_dataset--》{train_dataset}')
    # 过滤出text大于32的样本
    new_dataset = train_dataset.filter(lambda x: len(x["text"]) > 32)
    # print(f'new_dataset-->{new_dataset}')
    # print(f'new_dataset--》{len(new_dataset[0]["text"])}')
    # 实例化dataloader
    train_dataloader = DataLoader(new_dataset, batch_size=4, collate_fn=collate_fn2,
                                  shuffle=True, drop_last=True)

    for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
        print(f'input_ids--》{input_ids.shape}')
        print(f'attention_mask--》{attention_mask.shape}')
        print(f'labels--》{labels}')
        break

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # bert_tokenizer.vocab_size=21128
        self.linear = nn.Linear(768, bert_tokenizer.vocab_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 不更新预训练模型的参数
        with torch.no_grad():
            bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # print(f'bert_output--》{bert_output}')
        logits = bert_output.last_hidden_state[:, 16] # [8, 768]
        output = self.linear(logits) # [8, 21128]
        return output
def test_model():
    # 加载训练数据集
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # print(f'train_dataset--》{train_dataset[:3]}')
    # print(f'train_dataset--》{train_dataset}')
    # 过滤出text大于32的样本
    new_dataset = train_dataset.filter(lambda x: len(x["text"]) > 32)
    # print(f'new_dataset-->{new_dataset}')
    # print(f'new_dataset--》{len(new_dataset[0]["text"])}')
    # 实例化dataloader
    train_dataloader = DataLoader(new_dataset, batch_size=8, collate_fn=collate_fn2,
                                  shuffle=True, drop_last=True)

    my_model = MyModel()

    for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
        print('\n第1句mask的信息')
        print(bert_tokenizer.decode(input_ids[0]))
        print(bert_tokenizer.decode(labels[0]))
        print(bert_tokenizer.decode(input_ids[1]))
        print(bert_tokenizer.decode(labels[1]))
        output = my_model(input_ids, attention_mask, token_type_ids)
        print(f'output-->{output.shape}')
        break


def model2train():
    # 实例化模型
    my_model = MyModel()
    my_model = my_model.to(device)
    # 实例化优化器
    my_adamw = AdamW(my_model.parameters(), lr=5e-4)
    # 实例化损失函数
    my_crossentropy = nn.CrossEntropyLoss()
    # 实例化dataset数据源
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 32)
    # 不更新预训练模型的参数
    for param in bert_model.parameters():
        param.requires_grad_(False)

    # 设置模型为训练模型（因为使用预训练模型）
    my_model.train()

    # 设置训练的轮次
    epochs = 3

    # 开始训练
    for epoch_idx in range(1, epochs+1):
        # 实例化dataloader对象
        my_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn2,
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
            if i % 20 == 0:
                temp = torch.argmax(output, dim=-1)
                acc = (temp == labels).sum().item() / len(labels)
                print('轮次:%d 迭代数:%d 损失:%.6f 准确率%.3f 时间%d' \
                      %(epoch_idx, i, my_loss.item(), acc, (int)(time.time())-start_time))

        # 保存模型
        torch.save(my_model.state_dict(), './save_model/ai20_fill_mask_%d.bin' % epoch_idx)

def model2test():
    # 加载测试集dataset对象
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    test_dataset = test_dataset.filter(lambda x: len(x["text"]) > 32)
    # 加载训练好的模型
    my_model = MyModel()
    my_model.load_state_dict(torch.load('./save_model/ai20_fill_mask_3.bin', map_location='cpu'))

    # 实例化test_dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn2,
                                 drop_last=True, shuffle=True)

    # 设定模型为评估模式
    my_model.eval()
    # 设定几个变量
    correct = 0 # 预测正确的样本个数
    total = 0 # 已经迭代样本总个数

    # 开始测试
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(test_dataloader)):
        # input_ids-->[8, 32]
        with torch.no_grad():
            output = my_model(input_ids, attention_mask, token_type_ids)
        # print(f'output-->{output}')
        # 找出预测概率值对应的索引output-->[8, 21128]
        temp_idx = torch.argmax(output, dim=-1)
        # print(f'temp_idx--》{temp_idx}')
        correct = correct + (temp_idx == labels).sum().item()
        total = total + len(labels)

        if i % 25 == 0:
            print(correct / total, end='   ')
            # 抽取每个批次的第一个样本，来肉眼评估预测是否正确
            first_tokens = bert_tokenizer.decode(input_ids[0])
            print(first_tokens, end='   ')
            print('预测值：', bert_tokenizer.decode(temp_idx[0]), "真实值：", bert_tokenizer.decode(labels[0]))

if __name__ == '__main__':
    # test_model()
    # model2train()
    model2test()