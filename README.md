# HateTargetBERT-TR 

The use of hate speech targeting ethnicity, nationalities, religious identities, and specific groups has been on the rise in the news media. However, most existing automatic hate speech detection models focus on identifying hate speech, often neglecting the target group-specific language that is common in news articles. To address this problem, we first compile a hate speech dataset, **TurkishHatePrintCorpus**, derived from Turkish news articles and annotate it specifically for the language related to the targeted group. We then introduce the HateTargetBERT model, which integrates the target-centric linguistic features extracted in this study into the BERT model, and demonstrate its effectiveness in detecting hate speech while allowing the model's classification decision to be explained.


## How to Run

```bash
python src/train.py --name HateTargetNN --gpu_id 0 --batch_size 64 --num_workers 4 --epochs 10 --dataset_path .\data\turkishprintcorpus.csv --apply_preprocessing --include_linguistic_features --num_classes 2 --model_type HateTargetNN --optimizer_type AdamW --lr 1e-3
```

```bash
python src/train.py --name BERTurk --gpu_id 0 --batch_size 16 --num_workers 4 --epochs 3 --dataset_path  .\data\turkishprintcorpus.csv  --apply_preprocessing --num_classes 2 --model_type BERTurk --optimizer_type AdamW --lr 1e-5
```

```bash
python src/train.py --name HateTargetBERT --gpu_id 0 --batch_size 64 --num_workers 4 --epochs 3 --dataset_path .\data\turkishprintcorpus.csv --apply_preprocessing --include_linguistic_features --num_classes 2 --model_type HateTargetBERT --optimizer_type AdamW --lr 1e-5
```


### Pattern Ablation Study on HateTargetNN

```bash
python src/analyze_patterns.py --name HateTargetNN --gpu_id 0 --batch_size 64 --num_workers 4 --epochs 10 --dataset_path .\data\turkishprintcorpus.csv --apply_preprocessing --include_linguistic_features --num_classes 2 --model_type HateTargetNN --optimizer_type AdamW --lr 1e-3
```

