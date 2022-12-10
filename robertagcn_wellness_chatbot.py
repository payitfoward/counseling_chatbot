import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.utils.data as Data

from transformers import AutoModel, AutoTokenizer
from utils import *

import dgl

from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import os
import shutil
import argparse
import sys
import logging
from datetime import datetime

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss, Metric
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from model import BertGCN
import random

max_length = 64
batch_size = 256
ratio = 0.7
nb_epochs = 5
dataset = "wellness"
gcn_layers = 2
n_hidden = 200
heads = 8
dropout = 0.5
gcn_lr = 1e-3
bert_lr = 1e-5
args = [max_length, batch_size, ratio, nb_epochs, dataset, n_hidden, dropout, gcn_lr, bert_lr]

ckpt_dir = './checkpoint/robertagcn_{}_{}'.format(dataset, ratio)
os.makedirs(ckpt_dir, exist_ok=True)

streamhandle = logging.StreamHandler(sys.stdout)
streamhandle.setFormatter(logging.Formatter('%(message)s'))
streamhandle.setLevel(logging.INFO)

filehandle = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
filehandle.setFormatter(logging.Formatter('%(message)s'))
filehandle.setLevel(logging.INFO)

logger = logging.getLogger('training logger')
logger.addHandler(streamhandle)
logger.addHandler(filehandle)
logger.setLevel(logging.INFO)

cpu = torch.device('cpu')
gpu = torch.device('cuda:0')

logger.info('params:')
logger.info(str(args))
logger.info('checkpoints path: {}'.format(ckpt_dir))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)

nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]


model = BertGCN(nb_class=nb_class, ratio=ratio, n_hidden=n_hidden, dropout=dropout)

checkpoint = "./checkpoint/wellness/checkpoint.pth"
ckpt = torch.load(checkpoint, map_location=gpu)
model.bert_model.load_state_dict(ckpt['bert_model'])
model.classifier.load_state_dict(ckpt['classifier'])

corpse_file = './data/corpus/' + dataset +'_shuffle.txt'
with open(corpse_file, 'r', encoding="utf-8") as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')

def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    return input.input_ids, input.attention_mask


input_ids, attention_mask = encode_input(text, model.tokenizer)
input_ids = torch.cat([input_ids[:-nb_test], torch.zeros((nb_word, max_length), dtype=torch.long), input_ids[-nb_test:]])
attention_mask = torch.cat([attention_mask[:-nb_test], torch.zeros((nb_word, max_length), dtype=torch.long), attention_mask[-nb_test:]])

y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

doc_mask  = train_mask + val_mask + test_mask

adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    torch.LongTensor(y), torch.FloatTensor(train_mask), torch.FloatTensor(val_mask), torch.FloatTensor(test_mask)
g.ndata['label_train'] = torch.LongTensor(y_train)
g.ndata['cls_feats'] = torch.zeros((nb_node, model.feat_dim))

logger.info('graph information:')
logger.info(str(g))

train_idx = Data.TensorDataset(torch.arange(0, nb_train, dtype=torch.long))
val_idx = Data.TensorDataset(torch.arange(nb_train, nb_train + nb_val, dtype=torch.long))
test_idx = Data.TensorDataset(torch.arange(nb_node-nb_test, nb_node, dtype=torch.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

class F1Score(Metric):

    def __init__(self, *args, **kwargs):
        self.f1 = 0
        self.count = 0
        super().__init__(*args, **kwargs)

    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()

        _, predicted = torch.max(y_pred, 1)
        f = f1_score(y.cpu(), predicted.cpu(), average='macro')
        self.f1 += f
        self.count += 1

    def reset(self):
        self.f1 = 0
        self.count = 0
        super(F1Score, self).reset()

    def compute(self):
        return self.f1 / self.count
    
def update_feature():
    global model, g, doc_mask
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with torch.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = torch.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g


optimizer = torch.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.gcn.parameters(), 'lr': gcn_lr},
    ], lr=gcn_lr
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


def train_step(engine, batch):
    global model, g, optimizer
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(torch.BoolTensor)
    y_pred = model(g, idx)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with torch.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


trainer = Engine(train_step)
pbar = ProgressBar()
pbar.attach(trainer)

@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    torch.cuda.empty_cache()


def test_step(engine, batch):
    global model, g
    with torch.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true


evaluator = Engine(test_step)
eval_pbar = ProgressBar()
eval_pbar.attach(evaluator)
metrics={
    'acc': Accuracy(),
    'nll': Loss(torch.nn.NLLLoss()),
    'f1' : F1Score()
}
for name, function in metrics.items():
    function.attach(evaluator, name)
    
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll, val_f1 = metrics["acc"], metrics["nll"], metrics["f1"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll, test_f1 = metrics["acc"], metrics["nll"], metrics["f1"]
    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f} f1: {:.4f}  Test acc: {:.4f} loss: {:.4f} f1: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, 
                val_acc, val_nll, val_f1, 
                test_acc, test_nll, test_f1
               )
    )
    if test_acc > log_training_results.best_test_acc:
        logger.info("New checkpoint")
        torch.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
                'seed':trainer.state.seed,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_test_acc = test_acc

ckpt = torch.load(ckpt_dir + "/checkpoint.pth", map_location=gpu)
model.bert_model.load_state_dict(ckpt['bert_model'])
model.classifier.load_state_dict(ckpt['classifier'])
model.gcn.load_state_dict(ckpt['gcn'])

log_training_results.best_test_acc = 0
g = update_feature()
trainer.run(idx_loader, max_epochs=nb_epochs)

category, answer = load_chatbot_corpus()
test_index = len(g.ndata['input_ids']) - 1

while 1:
    text = input('\nQuestion: ')
    input_ids, attention_mask = encode_input(text, model.tokenizer)
    model.to(cpu)
    embedding = model.bert_model(input_ids, attention_mask)[0][:, 0]
    
    g.ndata['cls_feats'][test_index], g.ndata['input_ids'][test_index], g.ndata['attention_mask'][test_index] = embedding, input_ids, attention_mask
    model.eval()
    model = model.to(gpu)
    g = g.to(gpu)
    max_index = torch.argmax(model(g, torch.tensor([test_index]))).item()
    answer_list = answer[category[max_index]]
    answer_len = len(answer_list)-1
    answer_index = random.randint(0,answer_len)
    print(f'Answer: {answer_list[answer_index]}, index: {max_index}')
    print('-'*50)