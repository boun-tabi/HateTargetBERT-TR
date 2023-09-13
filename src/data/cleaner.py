import re

import cleantext
import langdetect
import pandas as pd
from zemberek import TurkishSentenceExtractor

CHAR_MAP = {
    "â": "a",
    "İ": "i",
    "I": "ı",
    "Â": "a",
    "Î": "i",
    "î": "i",
    "ì": "i",
    "í": "i",
    "û": "u",
    "Û": "u", 
    "|": "",
    "¦": "",
    "~": "",
    "°": "",
    "^": "",
    "+": "",
    "<": "",
    "=": "",
    ">": "",
    "±": "",
    "ĝ": "ğ",
    "ܧ": "",
    "'ާ": "",
    "ޱ": "",
    "绫": "",
    "ȥ": "",
    "ǥ": "",
    ". com. tr": ".com.tr",
    ". com": ".com",
    " km.": "km",
    "@hotmail. com": "@hotmail.com",
    " hotmail. com": "@hotmail.com"
}

EMAIL_PATTERNS = [
    "gmail",
    "hotmail",
    "yahoo",
    ".com",
]

class Cleaner():
    def __init__(self):
        self.extractor = TurkishSentenceExtractor()

    def __clean_text(self, text, clean_punc):
        if pd.isna(text):
            return text
        for char, replaced_char in CHAR_MAP.items():
            text = text.replace(char, replaced_char) 
        text = cleantext.clean(
            text,
            to_ascii=False,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=True,
            no_digits=True,
            no_currency_symbols=True,
            no_punct=clean_punc,
            no_emoji=True,
            replace_with_url=' ',
            replace_with_email=' ',
            replace_with_phone_number=' ',
            replace_with_punct='',
            replace_with_number='',
            replace_with_digit='',
            replace_with_currency_symbol=''
        )
        text = " ".join([token for token in text.split() if len(token) > 1])
        if not clean_punc:
            for email_token in EMAIL_PATTERNS:
                text = " ".join([token for token in text.split() if email_token not in token])
        return text
    
    def __remove_non_turkish_contents(self, df):
        df = df[df["content"] != ""]
        df["suggested_lang"] = df["content"].apply(lambda content: langdetect.detect(content))
        df = df[df["suggested_lang"] == "tr"]
        df = df.drop("suggested_lang", axis=1)
        return df
    
    def __split_sentences(self, text):
        # sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        sentences = self.extractor.from_paragraph(text)
        return sentences
    
    def __remove_punctuations(self, sentences):
        sentences = [self.__clean_text(sent, clean_punc=True) for sent in sentences]
        return sentences
    
    def __clean_bad_sentences(self, sentences):
        sentences = [sent for sent in sentences if len(sent.split()) > 1 and len(sent) >= 5]
        return sentences
    
    def process_df(self, df):
        for col in ["pub_name", "title", "content"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda text: self.__clean_text(text, clean_punc=False))
        df = self.__remove_non_turkish_contents(df)
        df["sentences"] = df["content"].apply(lambda text: self.__split_sentences(text))
        # df["sentences"] = df["sentences"].apply(lambda sentences: self.__remove_punctuations(sentences))
        df["sentences"] = df["sentences"].apply(lambda sentences: self.__clean_bad_sentences(sentences))
        df["text"] = df.apply(lambda row: " ".join(row["sentences"]).strip(), axis=1)
        df = df.drop("sentences", axis=1)
        return df
