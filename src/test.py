import argparse
import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data.hs_dataset import HateSpeechDataset
from models.hatetargetbert import HateTargetBERT
from models.hatetargetnn import HateTargetNN
from train import evaluate_model
from pathlib import Path
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

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--load_from', type=str, default='', help='load the pretrained model from the specified location')
parser.add_argument('--dataset_path', type=str, default='')
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

DATASET = args.dataset_path
LOAD_FROM = args.load_from
gpu_id = args.gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'

checkpoint = torch.load(args.load_from, map_location=f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')
config_file = checkpoint['config_file']
args = dotdict(json.load(open(config_file, 'r')))
if args.linguistic_features:
    pattern_types = args.linguistic_features.split(',') 
    rule_dimension = sum([pattern_dimensions[p] for p in pattern_types])
else:
    pattern_types = None
    rule_dimension = None

if args.model_type == "HateTargetNN":
    tokenizer = None
    model = HateTargetNN(num_labels=args.num_classes, rule_dimension=rule_dimension)

elif args.model_type == "HateTargetBERT":
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
    model = HateTargetBERT(checkpoint="dbmdz/bert-base-turkish-128k-uncased", num_labels=args.num_classes, rule_dimension=rule_dimension)
else: 
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-128k-uncased", num_labels=args.num_classes)

only_rules = args.model_type == "HateTargetNN"
test_dataset = HateSpeechDataset(split="test", 
                                 tokenizer=tokenizer, 
                                 data_path=args.dataset_path if DATASET == '' else DATASET, 
                                 apply_preprocessing=args.apply_preprocessing, 
                                 include_linguistic_features=args.include_linguistic_features, 
                                 only_rules=only_rules,
                                 pattern_types=pattern_types)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
if args.multigpu: 
    model = nn.DataParallel(model)
device = f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu'
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss()
test_metrics = evaluate_model(model, test_loader, criterion, device, args.model_type)

print(test_metrics)
with open(os.path.join(Path(LOAD_FROM).parent, "test_metrics.json"), "w") as f:
    json.dump(test_metrics, f)

# all_predictions = [output.argmax(dim=1).cpu().numpy() for output in test_metrics['outputs']]
# pred_df = test_dataset._get_prediction_results(all_predictions)
# pred_df.to_csv(os.path.join(args.load_from, "test_predictions.csv"), index=False)
