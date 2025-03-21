import re
import json
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification


#数据清洗
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[a-z\s]', '', text) #把字母和空格删掉
    return text

def filter_texts(df, min_length=4, max_length = 128):
    #把长度不符合要求的删掉
    df['char_count'] = df['text'].apply(lambda x: len([c for c in x if '\u4e00' <= c <= '\u9fff']))
    filtered_df = df[(min_length <= df['char_count']) & (df['char_count'] <= max_length)].drop(columns=['char_count'])
    return filtered_df


#数据输入
def input_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as text:
        for line in text:
            data.append(json.loads(line))
    return pd.DataFrame(data)   #变为pandas可读的数据格式


class CommentData(Dataset):
    def __init__(self, texts, points, tokenizer, max_len=128):
        self.texts = texts
        self.points = points
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        point = self.points[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,    #启用标识
            max_length=self.max_len,
            padding='max_length',       #不足补pad
            truncation=True,            #超出截断
            return_attention_mask=True, #掩码标记
            return_tensors='pt'         #返回张量
        )                               #训练方式

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(point, dtype=torch.float)
        }


class BertRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            local_model_path,  #用本地模型
            num_labels=1       #做回归任务
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        if labels is not None:
            loss = torch.nn.HuberLoss()(logits.squeeze(), labels)  #用Huberloss计算丢失值
            return loss, logits
        return logits


def training():
    patience = 5
    counter = 0
    best_epoch = 0

    train_df, test_df = train_test_split(df, test_size=0.2)
    train_dataset = CommentData(train_df['text'].values, train_df['point'].values, tokenizer, max_len=128)
    test_dataset = CommentData(test_df['text'].values, test_df['point'].values, tokenizer, max_len=128)  #分割数据
    model = BertRegressor()  #初始化bert

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  #学习速率参数调整
    evaluate_loss = torch.nn.MSELoss()  #丢失模型
    train_times = 10      #训练次数
    train_size = 12       #一次训练的样本数
    evaluate_size = 24    #一次评估的样本数
    update_frequency = 1  #每执行一次训练都更新数据

    train_dataloader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)
    eval_dataloader = DataLoader(test_dataset, batch_size=evaluate_size, shuffle=False)  #分别创建数据存储器
    device = torch.device("cpu")
    model.to(device)  #调用cpu训练模型

    scaler = torch.amp.GradScaler()  #处理梯度释放
    best_evalloss = float('inf')
    for epoch in range(train_times):
        model.train()
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch[
                'labels']  #加载数据
            optimizer.zero_grad()  #梯度置零

            with torch.amp.autocast('cpu', enabled=True):  #启用自动混合精度训练
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  #执行训练
                loss = outputs[0]  #收集丢失信息

            scaler.scale(loss).backward()  #反向传播避免梯度丢失
            scaler.step(optimizer)  #更新优化器确保梯度可用
            scaler.update()  #更新

        #测评
        model.eval()
        eval_loss = 0
        threshold = 0.8
        _preds = []
        _labels = []  #初始化变量
        with torch.no_grad():  #禁用梯度
            for batch in eval_dataloader:
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch[
                    'labels']

                with torch.amp.autocast('cpu', enabled=True):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs[0]
                    logits = outputs[1]  #收集预测相关数据

                logits = logits.to(torch.float32)
                batch_preds = logits.squeeze().numpy() * std + mean
                batch_labels = labels.numpy() * std + mean  #去归一化

                _preds.extend(batch_preds.tolist())
                _labels.extend(batch_labels.tolist())  #数据转换到列表中

                eval_loss += loss.item()  #记录丢失

        eval_loss /= len(eval_dataloader)
        print(f'\nEpoch: {epoch + 1}')
        print(f'Eval Loss: {eval_loss:.2f}')

        correct = sum(abs(np.array(_labels) - np.array(_preds)) <= threshold)
        accuracy = correct / len(_labels)
        print(f'Threshold Accuracy (±{threshold}): {accuracy:.2%}')

        print("\nSample Predictions:")
        for i in range(5):
            print(f" 真实评分: {_labels[i]:.1f}      预测评分: {_preds[i]:.1f}")

        #选择最佳模型
        if eval_loss < best_evalloss:
            best_evalloss = eval_loss
            counter = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), '../best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                print(f'best epoch is {best_epoch}')
                break

df = input_data('../comments_and_ratings.jsonl')
df['text'] = df['text'].apply(clean_text)
df = filter_texts(df)  #初始化评论信息

mean = df['point'].mean()
std = df['point'].std()
df['point'] = (df['point'] - mean) / std

local_model_path = "E:\\bert\\bert"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

training()

def predict(text, model, tokenizer, max_len=128, mean=mean, std=std):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        input_ids = encoding['input_ids'].to('cpu')
        attention_mask = encoding['attention_mask'].to('cpu')
        output = model(input_ids, attention_mask)

    predicted_score = (output[0].squeeze().item() * std) + mean  #去归一化
    return predicted_score
