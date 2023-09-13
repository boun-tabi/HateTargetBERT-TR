# import spacyturk

# # downloads the spaCyTurk model
# spacyturk.download("tr_floret_web_lg")

# # info about spaCyTurk installation and models
# spacyturk.info()

# load the model using spaCy
"""
Transformer based model pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_trf/resolve/main/tr_core_news_trf-any-py3-none-any.whl
Large model: pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_lg/resolve/main/tr_core_news_lg-any-py3-none-any.whl
Medium model: pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_md/resolve/main/tr_core_news_md-any-py3-none-any.whl
"""
import pandas as pd
from tqdm import tqdm
import spacy
from typing import Tuple, List, Union
tqdm.pandas()

def get_person(text, nlp) -> str:
    doc = nlp(text)
    if len(doc.ents) == 1 and doc.ents[0].label_ == 'PERSON':
        return doc.ents[0].text
    else:
        return ''


def extract_authors_from_titles(df: pd.DataFrame) -> pd.DataFrame:
    df['title'] = df['title'].fillna('')
    df['author_lg'] = df['title'].progress_apply(lambda t: get_person(t, lg_nlp))
    df['author_trf'] = df['title'].progress_apply(lambda t: get_person(t, trf_nlp))
    return df.drop_duplicates()


def load_authors_data(paths: list) -> Tuple[pd.DataFrame]:
    return tuple([pd.read_csv(path) for path in paths])


def replace_authors(row, authors, check_title=False) -> Tuple[str, list]:
    replaced_authors = []
    text = row["content"]
    for author in authors:
        if not check_title or author in row["title"]:
            if author in text:
                text = text.replace(author, '')
                replaced_authors.append(author)
    return text, replaced_authors


def update_dataframe_with_replacements(df: pd.DataFrame, authors_data: Tuple[pd.DataFrame]) -> pd.DataFrame:
    df['title'].fillna('', inplace=True)
    for a_df in authors_data:
        a_df['author_len'] = a_df['author'].apply(len) 
    frequent_authors = authors_data[0].sort_values(by='author_len', ascending=False)['author'].tolist()
    authors_from_content = authors_data[1].sort_values(by='author_len', ascending=False)['author'].tolist()
    authors_from_titles = authors_data[2].sort_values(by='author_len', ascending=False)['author'].tolist()

    for col_prefix, authors, check_title in [
        ('content_freq_title', frequent_authors, True),
        ('content_freq', frequent_authors, False),
        ('content_title', authors_from_titles, False),
        ('content_content', authors_from_content, False)
    ]:
        df[f'{col_prefix}'], df[f'replaced_authors_{col_prefix}'] = zip(*df.apply(lambda x: replace_authors(x, authors, check_title), axis=1))

    return df


def count_same_values(row) -> int:
    cols = ['content_freq_title', 'content_freq', 'content_title', 'content_content']
    unique_values = len(row[cols].unique())
    return 5 - unique_values


def most_frequent_value(row) -> str:
    cols = ['content_freq_title', 'content_freq', 'content_title', 'content_content']
    value_counts = row[cols].value_counts()
    return value_counts.index[0] if value_counts.iloc[0] >= 3 else row['content']


def replace_words(text, words_to_replace) -> str:
    for word in sorted(words_to_replace, key=len, reverse=True):
        text = text.replace(f'({word})', '')
    return text


def update_content(df: pd.DataFrame, agency_data: pd.DataFrame) -> pd.DataFrame:
    df['count_same_values'] = df.apply(count_same_values, axis=1)
    df['majority_value'] = df.apply(most_frequent_value, axis=1)
    df['final_content'] = df.apply(lambda x: replace_words(x['majority_value'], agency_data['agency'].tolist()), axis=1)
    return df
    
lg_nlp = spacy.load("tr_core_news_lg")
trf_nlp = spacy.load("tr_core_news_trf")


for path_in, path_out in [
        ('data/raw_data_v4_agg_filtered_v1.csv', 'data/raw_data_v4_agg_filtered_v1_author.csv'),
        ('data/raw_data_v4_agg_filtered_v2.csv', 'data/raw_data_v4_agg_filtered_v2_author.csv')
    ]:
    df = pd.read_csv(path_in)
    df = extract_authors_from_titles(df)
    df.to_csv(path_out, index=False)

# Manually review files and create authors.csv, authors_from_content.csv, authors_from_titles.csv

# Update dataframe with replacements
df = pd.read_csv('data/raw_data_v4_agg_filtered_v1.csv')
authors_data = load_authors_data(['data/authors.csv', 'data/authors_from_content.csv', 'data/authors_from_titles.csv'])
df = update_dataframe_with_replacements(df, authors_data)

# Update content
agency_data = pd.read_csv('data/agency.csv')
df = update_content(df, agency_data)
df.to_csv('data/raw_data_v4_agg_filtered_v1_updated.csv', index=False)