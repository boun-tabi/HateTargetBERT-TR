import pandas as pd
import numpy as np
import string
import kenlm
from zemberek import (
    TurkishSentenceNormalizer,
    TurkishSentenceExtractor,
    TurkishMorphology,
    TurkishTokenizer
)
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from pandarallel import pandarallel
from tqdm import tqdm
from typing import Tuple, List, Union

tqdm.pandas()

# Configuration Initializations
morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)
extractor = TurkishSentenceExtractor()
z_tokenizer = TurkishTokenizer.DEFAULT
v_tokenizer = PreTrainedTokenizerFast.from_pretrained('data/VBARTTokenizer')
model = kenlm.Model("data/tr_wiki_spiece_5gram.binary")

def score_sentence(sentence: str) -> float:
    sentence = sentence.lower().strip()
    tokens = v_tokenizer.tokenize(sentence)
    tokenized_sentence = " ".join(tokens)
    return model.score(tokenized_sentence, bos=True, eos=True)

def compute_sentence_statistics(text: str, word_score_threshold: float = -3.87407) -> Tuple[List[str], int, float, float, float, List[float], List[int], List[int]]:
    """
    Computes various statistics about sentences in the text.

    Args:
    text (str): The text to be analyzed.
    word_score_threshold (float, optional): Threshold for word score. Defaults to -3.87407.

    Returns:
    Tuple: Various computed statistics about the sentences in the text.
    """

    sentences = extractor.from_paragraph(text)
    num_sentences = len(sentences)
    if num_sentences == 0:
        return [], 0, 0, 0, 0, [], [], []

    avg_sentence_length_char = sum(len(sentence) for sentence in sentences) / num_sentences
    sentence_lengths_word = [len(z_tokenizer.tokenize(sentence)) for sentence in sentences]
    avg_sentence_length_word = sum(sentence_lengths_word) / num_sentences
    scores = [score_sentence(sentence) for sentence in sentences]
    
    weird_word_counts = []
    word_counts = []
    for sentence in sentences:
        sent = sentence.translate(str.maketrans('', '', string.punctuation))
        words = z_tokenizer.tokenize(sent)
        word_scores = [score_sentence(word.content) / (len(word.content) ** 0.66) for word in words]
        weird_word_counts.append(len([s for s in word_scores if s < word_score_threshold]))
        word_counts.append(len(words))
    
    return sentences, num_sentences, avg_sentence_length_char, avg_sentence_length_word, np.std(sentence_lengths_word), scores, word_counts, weird_word_counts

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the dataframe and computes statistics for each row content.

    Args:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Processed dataframe with additional statistics columns.
    """
    df = df.copy()
    df['type'] = df['type'].apply(lambda x: x.strip().lower())
    df['pub_name'] = df['pub_name'].apply(lambda x: x.strip().lower())

    computed_cols = df['content'].parallel_apply(lambda t: compute_sentence_statistics(t))
    df[["sentences", "num_sentences", "avg_sentence_length_char", "avg_sentence_length_word", "std_sentence_length_word", "scores", "token_counts", "weird_token_counts"]] = pd.DataFrame(computed_cols.tolist(), index=df.index)

    output_df = df[["id", "date", "pub_name", "type", "title", "phase", "Label", "content", "sentences", "num_sentences", "avg_sentence_length_char", "avg_sentence_length_word", "std_sentence_length_word", "scores", "token_counts", "weird_token_counts"]]
    output_df['tmp'] = output_df.apply(lambda row: list(zip(row["sentences"], row["scores"], row["token_counts"], row["weird_token_counts"])), axis=1) 
    output_df = output_df.explode("tmp")
    output_df[["sentences", "scores", "token_counts", "weird_token_counts"]] = pd.DataFrame(output_df["tmp"].tolist(), index=output_df.index)
    output_df.drop(columns='tmp', inplace=True)
    
    return output_df

def aggregate_and_filter_data(df: pd.DataFrame) -> pd.DataFrame:
    df['sentence_len'] = df['sentences'].apply(len)
    df['scores'] = df['scores'] / df['sentence_len']
    df['weird_ratio'] = df['weird_token_counts'] / df['token_counts']
    df['has_weird_words'] = df['weird_ratio'] > 0.25
    df['has_weird_sentences'] = df['scores'] < -1.9

    df_group = df.groupby('id').agg({
        'scores': [('scores_mean', 'mean'), ('scores_std', 'std')],
        'has_weird_words': [('has_weird_words_sum', 'sum')],
        'weird_token_counts': [('weird_token_counts_mean', 'mean')],
        'token_counts': [('token_counts_mean', 'mean')],
        'weird_ratio':  [('weird_ratio_mean', 'mean'), ('weird_ratio_max', 'max')],
        'has_weird_sentences': [('has_weird_sentences_sum', 'sum')]
    })

    output_df = pd.merge(df, df_group, on="id")
    output_df.drop_duplicates(['id'], inplace=True)
    return output_df

def save_data(df: pd.DataFrame):
    df.to_csv('data/raw_data_v4_agg.csv')

    filter_1 = (
        (df['has_weird_sentences_sum'] == 0) &
        (df['weird_ratio_mean'] < 0.2) &
        (df['weird_ratio_max'] < 0.5) &
        (df['weird_token_counts_mean'] <= 2) &
        (df['scores_mean'] > -0.94) &
        (df['scores_std'] < 0.2)
    )

    filter_2 = (
        (df['has_weird_sentences_sum'] == 0) &
        (df['weird_ratio_mean'] < 0.2) &
        (df['weird_ratio_max'] < 0.5) &
        (df['weird_token_counts_mean'] <= 2) &
        (df['scores_mean'] > -0.61) &
        (df['scores_std'] < 0.2)
    )

    df[filter_1].to_csv('data/raw_data_v4_agg_filtered_v1.csv', index=False)
    df[filter_2].to_csv('data/raw_data_v4_agg_filtered_v2.csv', index=False)

def load_sample_data(fraction: float = 0.01) -> pd.DataFrame:
    dataset = load_dataset("parquet", data_files={'train': 'data/data_cleaned_sentences_phases_rules.parquet'})
    df = dataset['train'].to_pandas()
    return df.sample(frac=fraction)

def generate_sentence_scores(sentences: List[str]) -> dict:
    sentence_scores = {}
    for sent in tqdm(sentences):
        sentence_scores[sent] = score_sentence(sent)
    return sentence_scores

def generate_word_scores(sentences: List[str]) -> dict:
    word_scores = {}
    for sent in sentences:
        sent = sent.translate(str.maketrans('', '', string.punctuation))
        words = z_tokenizer.tokenize(sent)
        for word in words:
            text = word.content
            if text not in word_scores:
                word_scores[text] = score_sentence(text)
    return word_scores


def save_dataframe_to_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def main():
    pandarallel.initialize(progress_bar=True)
    dataset = load_dataset("parquet", data_files={'train': 'data/data_cleaned_sentences_phases_rules.parquet'})
    df = dataset['train'].to_pandas()
    processed_df = process_dataframe(df)
    aggregated_df = aggregate_and_filter_data(processed_df)
    save_data(aggregated_df)

    pandarallel.initialize(progress_bar=True)
    sample_df = load_sample_data()
    corpus = ' '.join(sample_df['content'].tolist())
    sentences = extractor.from_paragraph(corpus)
    print('Number of sample sentences', len(sentences))

    word_scores = generate_word_scores(sentences)

    df = pd.DataFrame(list(word_scores.items()))
    df.columns = ['word', 'score']
    df['len'] = df['word'].apply(len)
    df['score_per_char'] = df['score'] / (df['len'] ** 0.66)
    df.to_csv('sample_word_scores.csv')



if __name__ == "__main__":
    main()