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
# 记载模型
bert_model = BertModel.from_pretrained('../dm08_transformers/bert-base-chinese')
bert_model = bert_model.to(device)
# print(f'bert_model--》{bert_model}')

# 加载dataset数据集
def dm_load_dataset():
    # 加载训练集
    # 默认如果不写split的话，返回结果是DatasetDict,加上split='train',返回一个dataset对象
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    print(f'train_dataset--》{train_dataset}')
    # print(f'train_dataset--》{train_dataset[0]}')
    print(f'train_dataset--》{train_dataset[:3]}')
    # 加载验证集
    # 默认如果不写split的话，返回结果是DatasetDict,加上split='train',返回一个dataset对象
    valid_dataset = load_dataset('csv', data_files='./data/validation.csv', split='train')
    print(f'valid_dataset--》{valid_dataset}')
    print(f'valid_dataset--》{valid_dataset[:3]}')

    # 加载测试集
    # 默认如果不写split的话，返回结果是DatasetDict,加上split='train',返回一个dataset对象
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    print(f'test_dataset--》{test_dataset}')
    print(f'test_dataset--》{test_dataset[:3]}')

def collate_fn1(data):
    sents = [i["text"] for i in data]
    labels = [i["label"] for i in data]
    # print(f'labels-->{labels}')
    # 对sents句子编码
    inputs = bert_tokenizer.batch_encode_plus(sents, truncation=True, padding="max_length",
                                              max_length=300, return_tensors='pt')
    # print(f'inputs--》{inputs}')

    input_ids = inputs["input_ids"] # inputs.input_ids
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, token_type_ids, labels
# 测试dataset
def dem02_test_dataset():
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
    # 对上述的dataset对象进行再次封装
    train_dataloader = DataLoader(train_dataset, batch_size=8,
                                  shuffle=True, drop_last=True,
                                  collate_fn=collate_fn1)

    # for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
    #     print(f'input_ids-->{input_ids.shape}')
    #     print(f'attention_mask-->{attention_mask.shape}')
    #     print(f'token_type_ids-->{token_type_ids.shape}')
    #     print(f'labels-->{labels}')
    #     break
    return train_dataloader


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

def dem03_test_model():
    my_model = MyModel()
    train_dataloader = dem02_test_dataset()
    for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
        print(f'input_ids-->{input_ids.shape}')
        print(f'attention_mask-->{attention_mask.shape}')
        print(f'token_type_ids-->{token_type_ids.shape}')
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
    # 实例化dataset数据源
    train_dataset = load_dataset('csv', data_files='./data/train.csv', split='train')
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
        my_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn1,
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
        torch.save(my_model.state_dict(), './save_model/ai20_text_class_%d.bin' % epoch_idx)


def model2test():
    # 加载测试集dataset对象
    test_dataset = load_dataset('csv', data_files='./data/test.csv', split='train')
    # 加载训练好的模型
    my_model = MyModel()
    my_model.load_state_dict(torch.load('./save_model/ai20_text_class_1.bin', map_location='cpu'))

    # 实例化test_dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn1,
                                 drop_last=True, shuffle=True)

    # 设定模型为评估模式
    my_model.eval()
    # 设定几个变量
    correct = 0 # 预测正确的样本个数
    total = 0 # 已经迭代样本总个数

    # 开始测试
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(tqdm(test_dataloader)):
        # input_ids-->[8, 300]
        with torch.no_grad():
            output = my_model(input_ids, attention_mask, token_type_ids)
        # print(f'output-->{output}')
        # 找出预测概率值对应的索引output-->[8, 2]
        temp_idx = torch.argmax(output, dim=-1)
        # print(f'temp_idx--》{temp_idx}')
        correct = correct + (temp_idx == labels).sum().item()
        total = total + len(labels)

        if i % 5 == 0:
            print(correct / total, end='   ')
            # 抽取每个批次的第一个样本，来肉眼评估预测是否正确
            first_tokens = bert_tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(first_tokens, end='   ')
            print('预测值：', temp_idx[0].item(), "真实值：", labels[0].item())




if __name__ == '__main__':
    # dm_load_dataset()
    # dem02_test_dataset()

    # dem03_test_model()
    # model2train()
    model2test()
