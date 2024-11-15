import argparse
import json
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
#from datasets import load_metric
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import utils
from data.hs_dataset import HateSpeechDataset
from models.hatetargetbert import HateTargetBERT
from models.hatetargetnn import HateTargetNN

pattern_dimensions = {
    'TS': 5,
    'TA': 1, 
    'MN': 5, 
    'HSI': 5,
    'PRE': 5,
    'POST': 5 
}

def init_args():
    parser = argparse.ArgumentParser()
    # Experiment Setup
    parser.add_argument('--name', type=str, default='imagenet2mnist', help='Name of the experiment. Determines where to store samples and models.')        
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID (e.g., -1, 0, 1, 2,). Use -1 for CPU.')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Directory to save models.')

    # Training Parameters 
    parser.add_argument('--batch_size', type=int, default=4, help='Input batch size.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data preparation.')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs.')
    parser.add_argument('--steps_for_eval', type=int, default=500, help='Number of steps for evaluation and model saving.')

    # Dataset Configuration
    parser.add_argument('--dataset_path', type=str, default='../data/data_cleaned_sentences_phases_2020-04-16.csv', help='Path to the dataset file.')  
    parser.add_argument('--apply_preprocessing', action='store_true', help='Enable preprocessing of the dataset before training.')  
    parser.add_argument('--include_linguistic_features', action='store_true', help='Include linguistic features')  
    parser.add_argument('--linguistic_features', type=str, default=None, help='Pattern type combination separated by commas (e.g. TA,MN,POST)')
    parser.add_argument('--num_classes', type=int, required=True,  help='Number of target classes in the dataset.') 

    # Model and Optimizer Settings
    parser.add_argument('--load_from', type=str, default='', help='Load the pretrained model from the specified location')
    parser.add_argument('--model_type', type=str, default='HateTargetNN', choices=['HateTargetNN', 'HateTargetBERT', 'BERTurk'], help='Model type.')
    parser.add_argument('--optimizer_type', type=str, default='AdamW', choices=["SGD", "Adam", "AdamW"], help='Optimizer type.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate for Adam optimizer.')

    return parser.parse_args()


def setup_data_loaders(dataset_path, apply_preprocessing, model_type, include_linguistic_features, only_rules, batch_size, num_workers, pattern_types=None):
    tokenizer = None
    if model_type != "HateTargetNN":
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")

    common_args = {
        'data_path': dataset_path,
        'apply_preprocessing': apply_preprocessing, 
        'include_linguistic_features': include_linguistic_features
    }

    train_dataset = HateSpeechDataset(split="train", tokenizer=tokenizer, **common_args, only_rules=only_rules, pattern_types=pattern_types)
    val_dataset = HateSpeechDataset(split="val", tokenizer=tokenizer, **common_args, only_rules=only_rules, pattern_types=pattern_types)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def setup_model_optimizer(model_type, num_classes, lr, optimizer_type='AdamW', rule_dimension=None, is_multigpu=False):
    if model_type == "HateTargetBERT":
        model = HateTargetBERT(checkpoint="dbmdz/bert-base-turkish-128k-uncased", num_labels=num_classes, rule_dimension=rule_dimension)
    elif model_type == "HateTargetNN":
        model = HateTargetNN(num_labels=num_classes, rule_dimension=rule_dimension)
    else:
        model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-128k-uncased", num_labels=num_classes)
    
    if is_multigpu:
        model = nn.DataParallel(model)

    # Optimizer
    optimizers = {
        "SGD": torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "Adam": torch.optim.Adam(model.parameters(), lr=lr),
        "AdamW": torch.optim.AdamW(model.parameters(), lr=lr)
    }

    optimizer = optimizers.get(optimizer_type)

    return model, optimizer

def train_epoch(epoch, model, train_loader, optimizer, criterion, device, model_type, steps_for_eval=500, val_loader=None, lr_scheduler=None):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    best_f1 = 0
    metric_data = []
    
    for i, (input_ids, attention_mask, label, rule) in enumerate(tqdm(train_loader)):

        label = label.to(device)
        if len(rule) > 0: 
            rule = rule.to(device)

        if len(input_ids) > 0:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if model_type == "HateTargetBERT":
                output = model(input_ids, attention_mask, label, rule)
            else:
                output = model(input_ids, attention_mask, labels=label)
            loss = output.loss
            logits = output.logits
        else:
            logits = model(rule)
            loss = criterion(logits, label)
                      
        if utils.check_loss(loss, loss.item()):
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        train_acc = (logits.argmax(dim=1) == label).float().mean()
        epoch_accuracy += train_acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

        if i % steps_for_eval == 0 and i != 0 and val_loader:
            val_metrics = evaluate_model(model, val_loader, criterion, device, model_type)
            if lr_scheduler:
                lr_scheduler.step(val_metrics['f1'])
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
            
            metric_data.append({
                "epoch": epoch,
                "step": i,
                "loss": epoch_loss,
                "acc": epoch_accuracy,
                "val_loss": val_metrics['loss'],
                "val_acc": val_metrics['accuracy'],
                "val_precision": val_metrics['precision'],
                "val_recall": val_metrics['recall'],
                "val_f1_score": val_metrics['f1']
            })

    return epoch_loss, epoch_accuracy, metric_data


def evaluate_model(model, val_loader, criterion, device, model_type):
    model.eval()
    
    val_loss = 0
    val_accuracy = 0
    val_outputs = []
    val_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, label, rule in val_loader:
            label = label.to(device)
            if len(rule) > 0: 
                rule = rule.to(device)

            if len(input_ids) > 0:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                if model_type == "HateTargetBERT":
                    output = model(input_ids, attention_mask, label, rule)
                else:
                    output = model(input_ids, attention_mask, labels=label)
                loss = output.loss
                logits = output.logits
            else:
                logits = model(rule)
                loss = criterion(logits, label)

            val_loss += loss.item()
            val_outputs.extend(logits.argmax(dim=1).cpu().numpy())
            val_labels.extend(label.cpu().numpy())
    
    val_accuracy = accuracy_score(val_labels, val_outputs)
    val_precision = precision_score(val_labels, val_outputs)
    val_recall = recall_score(val_labels, val_outputs)
    val_f1 = f1_score(val_labels, val_outputs)

    return {
        'loss': val_loss / len(val_loader),
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1, 
        'predictions': val_outputs,
        'labels': val_labels
    }


def setup_experiment(args):
    checkpoint_dir = os.path.join(args.checkpoints_dir, args.name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_uid = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    config_file = os.path.join(checkpoint_dir, f'config_{training_uid}.json')
    json.dump(vars(args), open(config_file, 'w'))
    utils.seed_everything(args.seed)

    return checkpoint_dir, training_uid, config_file

    
def main():
    args = init_args()
    checkpoint_dir, training_uid, config_file = setup_experiment(args)
    
    only_rules = args.model_type == "HateTargetNN"
    if args.linguistic_features:
        pattern_types = args.linguistic_features.split(',') 
        rule_dimension = sum([pattern_dimensions[p] for p in pattern_types])
    else:
        pattern_types = None
        rule_dimension = None

    train_loader, val_loader = setup_data_loaders(args.dataset_path, args.apply_preprocessing, args.model_type, 
                                                  args.include_linguistic_features, only_rules, args.batch_size, args.num_workers, pattern_types)


    device = 'cpu' if args.gpu_id == '-1' else f'cuda:{args.gpu_id}'
    model, optimizer = setup_model_optimizer(args.model_type, args.num_classes, args.lr, optimizer_type=args.optimizer_type, rule_dimension=rule_dimension)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, min_lr=1e-09, verbose=True)

    if args.load_from:
        checkpoint = torch.load(args.load_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['criterion']

    best_f1 = 0
    metric_df = pd.DataFrame(columns=["epoch", "step", "F1", "Accuracy", "Precision", "Recall"])


    for epoch in range(args.epochs):
        train_epoch(epoch, model, train_loader, optimizer, criterion, device, args.model_type, args.steps_for_eval, val_loader, lr_scheduler)
     
        val_metrics = evaluate_model(model, val_loader, criterion, device, args.model_type)

        if val_metrics['f1'] > best_f1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion,
                'config_file': config_file
            }, os.path.join(checkpoint_dir, f"best_model.pth"))
            best_f1 = val_metrics['f1']

        lr_scheduler.step(val_metrics['f1'])

        log_message = f"""Epoch : {epoch+1} - loss : {val_metrics['loss']:.4f} 
                        - val_acc: {val_metrics['accuracy']:.4f} - val_precision: {val_metrics['precision']:.4f}
                        - val_recall: {val_metrics['recall']:.4f} - val_f1_score: {val_metrics['f1']:.4f}\n"""

        with open(os.path.join(checkpoint_dir, f"training_log_{training_uid}.txt"), "a") as f:
            f.write(log_message)

        metric_df = metric_df.append({
            "epoch": epoch,
            "Precision": val_metrics['precision'],
            "Recall": val_metrics['recall'],
            "Accuracy": val_metrics['accuracy'],
            "F1": val_metrics['f1']
        }, ignore_index=True)

        metric_df.to_csv(os.path.join(checkpoint_dir, f"metrics_df_{training_uid}.csv"), index=False)

    metric_df.to_csv(os.path.join(checkpoint_dir, f"metrics_df_{training_uid}.csv"), index=False)


if __name__ == "__main__":
    main()





