import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import convent_feature
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
import os
import logging

EPOCHS = 4
BATCH_SIZE = 16
MAX_LEN = 200
LR = 5e-5
WARMUP_STEPS = 100
T_TOTAL = 10000 // BATCH_SIZE
DEVICE = 'cuda'
PRITN_STEPS = 20
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
logging.basicConfig(level=logging.INFO)

def load_feature(features, labels=None):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if labels is not None:
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, labels)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    return dataset


def load_data(tokenizer, max_seq):
    train = pd.read_csv("./data/train_new.csv", sep="\t")
    # test = pd.read_csv("./data/test_new.csv")
    train_words = train['comment'].values
    train_labels = train['label'].astype(int).values

    train_feature = convent_feature(train_words, tokenizer, max_seq)
    train_dataset = load_feature(train_feature, train_labels)
    return train_dataset


def batch_accuracy(pre, label):
    pre = pre.argmax(dim=1)
    correct = torch.eq(pre, label).sum().float().item()
    accuracy = correct / float(len(label))

    return accuracy


def train(model, dataset):
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    # optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=T_TOTAL)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=T_TOTAL
    )

    model.train()
    model.to(device=DEVICE)
    for _ in range(EPOCHS):
        epoch_iterator = tqdm(data_loader)
        t_loss = t_acc = 0.0
        for step, bath in enumerate(epoch_iterator):
            bath = tuple(t.to(DEVICE) for t in bath)
            inputs = {'input_ids': bath[0], "attention_mask": bath[1], "token_type_ids": bath[2], "labels": bath[3]}

            model.zero_grad()
            # optimizer.zero_grad()
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            loss.backward()
            scheduler.step()  # Update learning rate schedule
            optimizer.step()
            # model.zero_grad()

            t_loss += loss
            t_acc += batch_accuracy(logits, inputs["labels"])
            if (step + 1) % PRITN_STEPS == 0:
                t_loss /= PRITN_STEPS
                t_acc /= PRITN_STEPS
                print('\n' + '#'*100)
                print(f'\tepoch:{_ + 1} | step: {step + 1} | acc: {t_acc} | loss: {t_loss}')
                print('#' * 100)
                t_loss = t_acc = 0.0
                pre = logits.argmax(1)
                print(pre)
                print(inputs["labels"])


if __name__ == '__main__':
    model_path = '../model/chinese_wwm_ext_pytorch/'
    tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt')
    config = BertConfig.from_pretrained(model_path)
    config.num_labels = 2
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)

    train_dataset = load_data(tokenizer, max_seq=MAX_LEN)
    train(model, train_dataset)



