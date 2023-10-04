from sklearn.model_selection import train_test_split
import pandas as pd
OUTPUT_DIR = 'data'
random_state = 9
df = pd.read_csv(OUTPUT_DIR + '/TurkishHatePrintCorpus.csv')
train_val, test = train_test_split(df, test_size=0.10, random_state=random_state)
train, val = train_test_split(train_val, test_size=(1/9), random_state=random_state)
pd.concat([train.assign(split='train'), val.assign(split='val'), test.assign(split='test')]).to_csv(f'{OUTPUT_DIR}/TurkishHatePrintCorpus_run{random_state}.csv', index=False)

