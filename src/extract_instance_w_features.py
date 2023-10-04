from data.ling_feat_generator import LinguisticRuleGenerator
import pandas as pd
from pathlib import Path

path = Path('data')
for i in range(10): 
    data = pd.read_csv(path / f'TurkishHatePrintCorpus_run{i}.csv', sep=',')
    data = data[data['split'] == 'test']
    rule_assigner = LinguisticRuleGenerator('data/ling_feat_files')
    data =  rule_assigner.apply_rules(data, return_patterns=False)
    data['rules'] = data.apply(lambda x:  x['target_specific'] + x['target_agnostic'] + x['misleading_nonhateful'] + x['pre_target']+x['post_target'], axis=1) # x['hatespeech_indicators'] +
    data['all_zero'] = data['rules'].apply(lambda l: all(v == 0 for v in l))
    data[data['all_zero'] == False].to_csv(f'data/run{i}_test_features_only.csv')
    data[data['all_zero'] == False]
