import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from data.cleaner import Cleaner
from data.ling_feat_generator import LinguisticRuleGenerator

pattern_mapping = {
    'TS': 'target_specific',
    'TA': 'target_agnostic',
    'MN': 'misleading_nonhateful', 
    'HSI': 'hatespeech_indicators',
    'PRE': 'pre_target',
    'POST': 'post_target'
}

class HateSpeechDataset(Dataset):
    """
    Custom dataset object for processing hate speech data.
    """
    def __init__(
        self,
        split,
        tokenizer,
        data_path,
        apply_preprocessing=False,
        include_linguistic_features=False,
        only_rules=False, 
        pattern_types=None
    ):
        self.label_encodings = {"nonhateful": 0, "hateful": 1}
        self.tokenizer = tokenizer
        self.apply_preprocessing = apply_preprocessing
        self.include_linguistic_features = include_linguistic_features
        self.only_rules = only_rules
        self.split = split
        if not pattern_types:
            self.pattern_types = list(pattern_mapping.keys())
        else: 
            self.pattern_types = pattern_types
        self._process_dataset(data_path, split)
        
    def __len__(self):
        return len(self.labels)

    def _load_data(self, data_path, split):
        """
        Load and preprocess data.
        """
        # Load data
        data = pd.read_csv(data_path, sep=",")
        data = data[data["split"] == split]
        
        # Apply preprocessing if required
        # if self.apply_preprocessing:
        # text_cleaner = Cleaner()
        # data = text_cleaner.process_df(data)
        # else:
        #     data["title"] = data["title"].apply(lambda title: title if isinstance(title, str) else "")
        #     data["text"] = data.apply(lambda row: " ".join(row["sentences"]), axis=1)
        
        # Add linguistic features if required
        if self.include_linguistic_features:
            rule_assigner = LinguisticRuleGenerator()
            data = rule_assigner.apply_rules(data)
            self._add_linguistic_features(data)
        
        # Drop unused columns
        data = data.drop("title", axis=1)
        
        # Map labels to encodings
        data["Label"] = data["Label"].map(self.label_encodings)
        
        return data

    def _add_linguistic_features(self, data):
        """
        Combine linguistic features and drop individual columns.
        """
        data["all_rules"] = data.apply(lambda row: np.array(
            [item for col_key in self.pattern_types for item in row[pattern_mapping[col_key]]]
        ).astype(np.float32), axis=1)

        columns_to_drop = ["target_specific", "target_agnostic", "misleading_nonhateful", "hatespeech_indicators", "pre_target", "post_target"]
        columns_to_drop += [f'{col}_spans' for col in columns_to_drop if f'{col}_spans' in data.columns]
        data.drop(columns_to_drop, axis=1, inplace=True)

    def _process_dataset(self, data_path, split, pattern_types):
        """
        Process data for training/validation.
        """
        data = self._load_data(data_path, split, pattern_types)
        self.labels = list(data["Label"].values)
        self.idxs = list(data["id"].values)
        if self.include_linguistic_features:
            self.rules = list(data["all_rules"].values)
        if self.tokenizer:
            instances = self.tokenizer(data['text'].tolist(), truncation=True, padding=True)
            self.input_ids = instances['input_ids']
            self.attention_masks = instances['attention_mask']


    def __getitem__(self, idx):
        """
        Get item from dataset by index.
        """
        label = self.labels[idx]

        rule = []
        if self.include_linguistic_features:
            rule = np.array(self.rules[idx])

        # If only rules are needed
        if self.only_rules:
            return [], [], label, rule
        
        input_id = np.array(self.input_ids[idx])
        attention_mask = np.array(self.attention_masks[idx])
        return input_id, attention_mask, label, rule


    def get_prediction_results(self, preds):
        """
        Return predictions as a dataframe.
        """
        df = pd.DataFrame(data={
            "id": self.idxs,
            "prediction": preds,
            "label": self.labels
        })
        return df
