import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import convent_feature
import time
import random
import pandas as pd
import os
import logging
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

SUM = 10000
EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 200
LR = 1e-5
WARMUP_STEPS = 100
T_TOTAL = SUM // BATCH_SIZE
DEVICE = 'cuda'
PRINT_STEPS = 20
TRAIN_PROPORTION = 0.9
EARLY_STOP = 6 # loss not change stop
RESAMPLE = True
WEIGHT = 3.0

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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


def load_train_data(tokenizer, max_seq):
    train_file = pd.read_csv("./data/train_new.csv", sep="\t")
    train_words = train_file['comment'].values
    train_labels = train_file['label'].astype(int).values

    train_feature = convent_feature(train_words, tokenizer, max_seq)
    # train_dataset = load_feature(train_feature, train_labels)
    return train_feature, train_labels


def load_test_data(tokenizer, max_seq):
    test_file = pd.read_csv("./data/test_new.csv")
    test_words = test_file['comment'].values

    test_feature = convent_feature(test_words, tokenizer, max_seq)
    test_dataset = load_feature(test_feature)
    return test_dataset


def batch_accuracy(pre, label):
    pre = pre.argmax(dim=1)
    correct = torch.eq(pre, label).sum().float().item()
    accuracy = correct / float(len(label))

    return accuracy


# def data_split(dataset, split):
#     # indexs =list(range(SUM))
#     # random.shuffle(indexs)
#     # dataset = [dataset[i] for i in indexs]
#     return TensorDataset(*dataset[:split]), TensorDataset(*dataset[split:])


def train(model, train_data, dev_data, fold):
    # print(len(train_data))
    # bert_cls_1.0-propotion：2.0
    train_weight = [WEIGHT if data[-1] == 1 else 1.0 for data in train_data]
    if RESAMPLE:
        sample = WeightedRandomSampler(train_weight, num_samples=SUM)
    else:
        sample = RandomSampler(train_data)
    train_loader = DataLoader(dataset=train_data, sampler=sample, batch_size=BATCH_SIZE)
    dev_loader = DataLoader(dataset=dev_data, batch_size=BATCH_SIZE * 4, shuffle=False)
    # print(dev_loader)

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
    best_f1 = pre_f1 = 0.0
    t_loss = t_acc = 0
    for _ in range(EPOCHS):
        t_step = 0
        epoch_iterator = tqdm(train_loader)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(DEVICE) for t in batch)
            inputs = {'input_ids': batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                      "labels": batch[3]}

            model.zero_grad()
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            t_loss += loss
            t_acc += batch_accuracy(logits, inputs["labels"])

            if (step + 1) % PRINT_STEPS == 0:
                # print('\n', f'loss:{t_loss / (step + 1)} | acc: {t_acc / (step + 1)}')
                # print(logits.argmax(1))
                # print(inputs["labels"])
                precision, recall, f1, acc, loss = eval(model, dev_loader)
                print('\n' + "#"*100)
                logging.info(f'\tfold: {fold} | epoch: {_}')
                logging.info(f'\tprecision: {precision} | recall: {recall}| F1: {f1}')
                logging.info(f'\tacc: {acc}| loss: {loss}| train_loss: {t_loss / (step + 1)}')
                print("#" * 100)
                # EARLY_STOP
                if abs(pre_f1 - f1) < 1e-6:
                    t_step += 1
                    if t_step == EARLY_STOP:
                        logging.debug("****************EARLY STOP!****************************")
                        break
                else:
                    t_step = 0
                print(f"############ t_sep: {t_step}##############")
                if f1 > best_f1:
                    torch.save(model.state_dict(), './outputs/bert_cla_{}_{:.5f}.ckpt'.format(fold, f1))
                    if os.path.exists('./outputs/bert_cla_{}_{:.5f}.ckpt'.format(fold, best_f1)):
                        os.remove('./outputs/bert_cla_{}_{:.5f}.ckpt'.format(fold, best_f1))
                    logging.info("GEST THE BETTER MODEL")
                    best_f1 = f1
                pre_f1 = f1

    return best_f1


def eval(model, data_loader):
    logging.info("\n\n****EVALUATE START****")
    t_loss = step = 0
    pre = []
    labels = []
    model.eval()
    model.cuda()
    # s_time = time.time()
    for step, batch in enumerate(data_loader):
        with torch.no_grad():
            batch = tuple(t.to(DEVICE) for t in batch)
            inputs = {'input_ids': batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                      "labels": batch[3]}
            outputs = model(**inputs)
        loss, logits = outputs[:2]
        t_loss += loss
        pre.append(logits.argmax(1)), labels.append(inputs['labels'])
    pre, labels = torch.cat(pre), torch.cat(labels)
    # t_time = time.time()
    # print(t_time - s_time)
    TP = ((pre == 1) & (labels == 1)).sum().item()
    TN = ((pre == 0) & (labels == 0)).sum().item()
    FN = ((pre == 0) & (labels == 1)).sum().item()
    FP = ((pre == 1) & (labels == 0)).sum().item()

    eci = 1e-10
    p, r = TP / float(TP + FP + eci), TP / float(TP + FN + eci)
    f1, acc = 2 * r * p / (p + r + eci), (TP + TN) / (TP + TN + FP + FN)
    logging.info("****EVALUATE END****")
    # print(time.time() - t_time)
    return p, r, f1, acc, t_loss / (step + 1)


def test(path, model, dataset):
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.load_state_dict(torch.load(path))
    logging.info("****TEST START****")
    labels = []
    model.eval()
    model.cuda()
    for step, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            batch = tuple(t.to(DEVICE) for t in batch)
            inputs = {'input_ids': batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            outputs = model(**inputs)
        logits = outputs[0]
        labels.append(logits)
    labels = torch.cat(labels)
    logging.info("****TEST END****")
    return labels.softmax(dim=1).cpu()


if __name__ == '__main__':
    # model_path = 'D:\\model\\chinese_wwm_ext_pytorch\\'
    model_path = '/home/mhl/model/chinese_wwm_ext_pytorch/'
    tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt')
    config = BertConfig.from_pretrained(model_path)
    config.num_labels = 2
    #
    features, labels = load_train_data(tokenizer, max_seq=MAX_LEN)
    test_file = pd.read_csv("./data/test_new.csv")
    test_dataset = load_test_data(tokenizer, max_seq=MAX_LEN)

    max_sor = 0
    flag = torch.zeros(len(test_file), 2)
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    # for fold, (train_index, valid_index) in enumerate(kfold.split(features, labels)):
    #     train_x = [features[i] for i in train_index]
    #     train_y = labels[train_index]
    #     train_dataset = load_feature(train_x, train_y)
    #     valid_x = [features[i] for i in valid_index]
    #     valid_y = labels[valid_index]
    #     valid_dataset = load_feature(valid_x, valid_y)
    #
    #     model = BertForSequenceClassification.from_pretrained(model_path, config=config)
    #     sorce = train(model, train_dataset, valid_dataset, fold)
    #     flag += test('./outputs/bert_cla_{}_{:.5f}.ckpt'.format(fold, sorce), model, test_dataset)
    #     max_sor = max(sorce, max_sor)
    fold, sorce = 1, 0.95008
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)
    flag += test('./outputs/bert_cla_{}_{:.5f}.ckpt'.format(fold, sorce), model, test_dataset)
    test_file['flag'] = flag.argmax(dim=1).cpu().numpy().tolist()
    test_file[['id', 'flag']].to_csv('./result/my_bert_{:.5f}.csv'.format(max_sor), index=False)

