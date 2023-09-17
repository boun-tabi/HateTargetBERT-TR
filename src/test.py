import argparse
import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer
from data.hs_dataset import HateSpeechDataset
from models.hatetargetbert import HateTargetBERT
from models.hatetargetnn import HateTargetNN
from train import evaluate_model

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__        

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--load_from', type=str, default='', help='load the pretrained model from the specified location')
args = parser.parse_args()

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'

checkpoint = torch.load(args.load_from, map_location=f'cuda:{gpu_id}')
config_file = checkpoint['config_file']
args = dotdict(json.load(open(config_file, 'r')))

if args.model_type == "HateTargetNN":
    tokenizer = None
    model = HateTargetNN(num_labels=args.num_classes, rule_dimension=26)

else:
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
    model = HateTargetBERT(checkpoint="dbmdz/bert-base-turkish-128k-uncased", num_labels=args.num_classes, rule_dimension=26)

only_rules = args.model_type == "HateTargetNN"
test_dataset = HateSpeechDataset(split="test", 
                                 tokenizer=tokenizer, 
                                 data_path=args.dataset_path, 
                                 apply_preprocessing=args.apply_preprocessing, 
                                 include_linguistic_features=args.include_linguistic_features, 
                                 only_rules=only_rules)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
model = nn.DataParallel(model)
device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss()
test_metrics = evaluate_model(model, test_loader, criterion, device, args)

print(test_metrics)
with open(os.path.join(args.load_from, "test_metrics.json"), "w") as f:
    json.dump(test_metrics, f)

all_predictions = [output.argmax(dim=1).cpu().numpy() for output in test_metrics['outputs']]
pred_df = test_dataset._get_prediction_results(all_predictions)
pred_df.to_csv(os.path.join(args.load_from, "test_predictions.csv"), index=False)