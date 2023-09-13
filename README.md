# HateTargetBERT-TR 

## Requirements

```
clean-text
datasets
langdetect
numpy
pandas
scikit-learn
spacy
spacyturk
torch 
transformers
tqdm
zemberek-python
```

## How to Run

```
python src/train.py --name HateTargetNN --gpu_id 0 --batch_size 64 --num_workers 4 --epochs 10 --dataset_path .\data\turkishprintcorpus.csv --max_sentence_length 200 --max_sentences_per_article 50 --apply_preprocessing --include_linguistic_features --num_classes 2 --model_type HateTargetNN --optimizer_type AdamW --lr 1e-5
```

## Citation

