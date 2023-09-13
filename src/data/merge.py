from datasets import load_dataset
import pandas as pd


dataset = load_dataset("parquet", data_files={'train': 'data/data_cleaned_sentences_phases_rules.parquet'})
df = dataset['train'].to_pandas()
print(df.columns)

df_v1 = pd.read_csv('data/raw_data_v4_agg_filtered_v1_updated.csv')
df_v2 = pd.read_csv('data/raw_data_v4_agg_filtered_v2_updated.csv')

print(df_v1.columns)
cols = ['sentences','text', 'special_pattern','general_rule',	'anti_hs',	'hs_specific_verb',	'adj_bef_keyword',	'adj_after_keyword']
pd.merge(df, df_v1[["id", "final_content"]], on=['id'], how='inner').drop(columns=cols).to_csv('data/raw_data_v4_agg_filtered_v1_updated_labels.csv', index=False)
pd.merge(df, df_v2[["id", "final_content"]], on=['id'], how='inner').drop(columns=cols).to_csv('data/raw_data_v4_agg_filtered_v2_updated_labels.csv', index=False)
