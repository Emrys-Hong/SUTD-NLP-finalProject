from utils import NERDataset
from utils import collate_cloze_test
from functools import partial
from config import Config
from typing import List
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizer
from tqdm import tqdm
from utils import get_optimizers, write_data, read_data
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, classification_report
import os
import random
import json
from copy import deepcopy
import argparse
from conlleval import evaluate1


conf = Config()
conf.labels = list(set([j for i in read_data(conf.train_file, 1) for j in i]))
conf.labels.remove('O')
conf.labels = ['O'] + conf.labels
conf.label_map = {k:v for v, k in enumerate(conf.labels)}
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser: argparse.ArgumentParser):
    # training
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])


    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train(config: Config, train_dataloader: DataLoader, num_epochs: int,
          bert_model_name: str, num_labels: int,
          dev: torch.device, valid_dataloader: DataLoader = None, labels_list: List = None):

    gradient_accumulation_steps = 1
    t_total = int(len(train_dataloader) // gradient_accumulation_steps * num_epochs)

    model = BertForTokenClassification.from_pretrained(bert_model_name)
    model.classifier = nn.Linear(768, num_labels)
    model = model.to(dev)

    optimizer, scheduler = get_optimizers(config, model, t_total)
    model.zero_grad()
    loss_function = nn.CrossEntropyLoss().to(dev)

    best_performance = -1
    best_weight = deepcopy(model.state_dict())
    os.makedirs(f"model_files/{config.model_folder}", exist_ok=True)
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for iter, feature in tqdm(enumerate(train_dataloader, 1) , desc="--training batch", total=len(train_dataloader)):
            optimizer.zero_grad()
            preds = model(input_ids=feature.input_ids.to(dev), token_type_ids=feature.token_type_ids.to(dev), attention_mask=feature.attention_mask.to(dev))[0]
            loss = loss_function(preds.permute(0, 2, 1), feature.label_id.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if iter % 1000 == 0:
                print(f"epoch: {epoch}, iteration: {iter}, current mean loss: {total_loss/iter:.2f}", flush=True)
        print(f"Finish epoch: {epoch}, loss: {total_loss:.2f}, mean loss: {total_loss/len(train_dataloader):.2f}", flush=True)
        if valid_dataloader is not None:
            print("[Model Info] Evaluation: \n")
            performance = evaluate(valid_dataloader, model, dev)
            if performance > best_performance:
                print(f"[Model Info] Saving Model...")
                best_performance = performance
                best_weight = deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"model_files/{config.model_folder}/model_{round(performance, 4)}.pt")
    print(f"[Model Info] Saving Best Model...")
    torch.save(best_weight, f"model_files/{config.model_folder}/model_best.pt")
    print(f"[Model Info] Best validation performance: {best_performance}")
    return model


def evaluate(valid_dataloader: DataLoader, model: nn.Module, dev: torch.device,
             print_error:bool = False, error_file: str = None, labels_list: List = None) -> float:
    model.eval()
    predictions = []
    labels = []
    predicted_prob = []
    with torch.no_grad():
        for index, feature in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            preds = model(input_ids = feature.input_ids.to(dev), token_type_ids=feature.token_type_ids.to(dev), attention_mask=feature.attention_mask.to(dev))[0]
            preds = preds.softmax(dim=-1)
            preds = preds.cpu().numpy()
            current_prediction = np.argmax(preds, axis=-1)
            predictions.append(current_prediction.flatten())
            labels.append(feature.label_id.cpu().numpy().flatten())
    y_label = [conf.labels[oo] for o in labels for oo in o+[0]]
    y_preds = [conf.labels[oo] for o in predictions for oo in o+[0]]
    
    prec, rec, f1 = evaluate1(y_label, y_preds, verbose=False)
    print(f'precision: {prec:.3f} \t rec: {rec:.3f} \t f1 {f1:.3f}')
    return f1

def main():
    parser = argparse.ArgumentParser(description="NLP NER")
    opt = parse_arguments(parser)
    set_seed(conf.seed)

    bert_model_name = conf.bert_model_name
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    label_map = conf.label_map
    labels = conf.labels

    # Read dataset
    if opt.mode == "train":
        print("[Data Info] Reading training data", flush=True)
        dataset = NERDataset(file=conf.train_file, tokenizer=tokenizer, label_map=label_map)
        print("[Data Info] Reading validation data", flush=True)
        eval_dataset = NERDataset(file=conf.dev_file, tokenizer=tokenizer, label_map=label_map)

        # Prepare data loader
        print("[Data Info] Loading training data", flush=True)
        train_dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=conf.shuffle_train_data, num_workers=conf.num_workers, collate_fn=partial(collate_cloze_test, tokenizer=tokenizer))
        print("[Data Info] Loading validation data", flush=True)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=partial(collate_cloze_test, tokenizer=tokenizer))

        # Train the model
        model = train(conf, train_dataloader,
                      num_epochs= conf.num_epochs,
                      bert_model_name= bert_model_name,
                      valid_dataloader= valid_dataloader,
                      dev=conf.device,
                      num_labels=len(label_map))
    else:
        print(f"Testing the model now.")
        model = BertForTokenClassification.from_pretrained(bert_model_name)
        model.classifier = nn.Linear(768, len(label_map))
        model.load_state_dict(torch.load(f"model_files/{conf.model_folder}/model_best.pt", map_location=conf.device))
        model = model.to(conf.device)
        print("[Data Info] Reading test data", flush=True)
        eval_dataset = NERDataset(file=conf.dev_file, tokenizer=tokenizer, label_map=label_map)

        print("[Data Info] Loading validation data", flush=True)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=partial(collate_cloze_test, tokenizer=tokenizer))
        evaluate(valid_dataloader, model, conf.device, labels_list=labels)


if __name__ == "__main__":
    main()
