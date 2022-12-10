import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.utils.data as Data

from transformers import AutoModel, AutoTokenizer
from utils import *

import dgl

import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score

import os
import sys
import argparse
import shutil
import logging

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from model import BertClassifier

max_length = 128
batch_size = 128
nb_epochs = 30
bert_lr = 1e-4
dataset = "wellness"
ckpt_dir = './checkpoint/{}'.format(dataset)

args = [max_length, batch_size, nb_epochs, dataset, bert_lr]
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
gpu = torch.device('cuda:1')

logger.info('params:')
logger.info(str(args))
logger.info('checkpoints path: {}'.format(ckpt_dir))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)

nb_node = adj.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

model = BertClassifier(nb_class=nb_class)

y = torch.LongTensor((y_train + y_val +y_test).argmax(axis=1))
label = {}
label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train+nb_val], y[-nb_test:]

corpus_file = './data/corpus/'+dataset+'_shuffle.txt'
with open(corpus_file, 'r', encoding="utf-8") as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')

def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
    return input.input_ids, input.attention_mask

input_ids, attention_mask = {}, {}

input_ids_, attention_mask_ = encode_input(text, model.tokenizer)

input_ids['train'], input_ids['val'], input_ids['test'] =  input_ids_[:nb_train], input_ids_[nb_train:nb_train+nb_val], input_ids_[-nb_test:]
attention_mask['train'], attention_mask['val'], attention_mask['test'] =  attention_mask_[:nb_train], attention_mask_[nb_train:nb_train+nb_val], attention_mask_[-nb_test:]

datasets = {}
loader = {}
for split in ['train', 'val', 'test']:
    datasets[split] =  Data.TensorDataset(input_ids[split], attention_mask[split], label[split])
    loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True)



optimizer = torch.optim.Adam(model.parameters(), lr=bert_lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


def train_step(engine, batch):
    global model, optimizer
    model.train()
    model = model.to(gpu)
    optimizer.zero_grad()
    (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    y_pred = model(input_ids, attention_mask)
    y_true = label.type(torch.long)
    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    with torch.no_grad():
        y_true = y_true.detach().cpu()
        y_pred = y_pred.argmax(axis=1).detach().cpu()
        train_acc = accuracy_score(y_true, y_pred)
    return train_loss, train_acc


trainer = Engine(train_step)
pbar = ProgressBar()
pbar.attach(trainer)


def test_step(engine, batch):
    global model
    with torch.no_grad():
        model.eval()
        model = model.to(gpu)
        (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
        optimizer.zero_grad()
        y_pred = model(input_ids, attention_mask)
        y_true = label
        return y_pred, y_true


evaluator = Engine(test_step)
eval_pbar = ProgressBar()
eval_pbar.attach(evaluator)
metrics={
    'acc': Accuracy(),
    'nll': Loss(torch.nn.CrossEntropyLoss())
}
for name, function in metrics.items():
    function.attach(evaluator, name)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(loader['train'])
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(loader['val'])
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(loader['test'])
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    logger.info(
        "\rEpoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_nll)
    )
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        torch.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc
    scheduler.step()

        
log_training_results.best_val_acc = 0
trainer.run(loader['train'], max_epochs=nb_epochs)