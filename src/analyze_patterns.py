import os
import json
import pandas as pd
import torch 
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import combinations
from train import init_args, setup_experiment, setup_data_loaders, setup_model_optimizer, train_epoch, evaluate_model
from datetime import datetime
from data.hs_dataset import HateSpeechDataset
from models.hatetargetnn import HateTargetNN

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__   

pattern_dimensions = {
    'TS': 5,
    'TA': 1, 
    'MN': 5, 
    'HSI': 5,
    'PRE': 5,
    'POST': 5 
}

def run_experiment(pattern_types, args):
    name = args.name + '_' + '_'.join(pattern_types)
    rule_dimension = sum([pattern_dimensions[p] for p in pattern_types])

    checkpoint_dir = os.path.join(args.checkpoints_dir, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_uid = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    config_file = os.path.join(checkpoint_dir, f'config_{training_uid}.json')
    only_rules = True
    train_loader, val_loader = setup_data_loaders(args.dataset_path, args.apply_preprocessing, args.model_type, 
                                                  args.include_linguistic_features, only_rules, args.batch_size, 
                                                  args.num_workers, pattern_types)


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
        train_epoch(epoch, model, train_loader, optimizer, criterion, device, args.model_type, args.step_for_eval, val_loader, lr_scheduler)
     
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
        
    checkpoint = torch.load(os.path.join(checkpoint_dir, f"best_model.pth"), map_location=device)
    config_file = checkpoint['config_file']
    args = dotdict(json.load(open(config_file, 'r')))

    model = HateTargetNN(num_labels=args.num_classes, rule_dimension=rule_dimension)
    test_dataset = HateSpeechDataset(split="test", 
                                    tokenizer=None, 
                                    data_path=args.dataset_path, 
                                    apply_preprocessing=args.apply_preprocessing, 
                                    include_linguistic_features=args.include_linguistic_features, 
                                    only_rules=only_rules, 
                                    pattern_types=pattern_types)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_model(model, test_loader, criterion, device, args.model_type)

    print(test_metrics)
    with open(os.path.join(checkpoint_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f)


def main():
    args = init_args()
    pattern_list = ['TA', 'TS', 'PRE', 'POST', 'HSI', 'MN']
    for i in range(1, 6): 
        patterns = list(combinations(pattern_list, i))
        for pattern in patterns: 
            run_experiment(pattern, args)



if __name__ == "__main__":
    main()
