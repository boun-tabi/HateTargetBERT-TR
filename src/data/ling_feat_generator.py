import json
import re

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

class LinguisticRuleGenerator:
    def __init__(self, path="data/ling_feat_files/"):
        self._initialize_rules()
        self._load_external_data(path)
        self._prepare_rules()

    def _initialize_rules(self):
        self.target_agnostic_rules = [        
            r"\s[IRK](?!istan|stan|ya)(lık|lik|luk|lük)\s(.*?)(yap)",
            r"\s[IRK_KIMLIK](?!istan|stan|ya)\s(.*?)(skandalı)",
            r"\s[KIMLIK](?!istan|stan|ya)(ların|lerin|in|ın|un|ün)?\s(.*?)(işgali)",
            r"\s[IRK_KIMLIK](?!istan|stan|ya)(a|e)\s(bak sen)",
            r"\s[IRK](?!istan|stan|ya)(ların|lerin|in|ın|un|ün)?\s(.*?)(işgali)",
            r"\s[IRK](?!istan|stan|ya)\s(.*?)\s[IRK](?!istan|stan|ya)(lığını|liğini|luğunu|lüğünü)\s(.*?)(yap)",
            r"\s[IRK](?!istan|stan|ya)(ın|in|un|ün)\s(.*?)(uşağı|işbirlikçisi|piyonu|kuklası)(.*?)\s[IRK](?!istan|stan|ya)\s",
            r"\s[IRK](?!istan|stan|ya)(ın|in|un|ün)\s(.*?)(hain|vahşi|insanlık dışı|hunharca|kan donduran|şeytani|sinsi|[ADJBEF])\s(teşebbüsleri|planları|oluşumları|[ADJAFTER])",
            r"\s[IRK](?!istan|stan|ya)\s(.*?)(kurşunlarıyla|bombalarıyla|parasıyla)",
            r"\s[IRK](?!istan|stan|ya)(ın|in|un|ün)\s(.*?)(gerçekleştirdiği|yaptığı)(.*?)(katliam|zulüm|soykırım|[ADJAFTER])",
            r"\s[IRK](?!istan|stan|ya)\s(tarafından)\s(gerçekleştirilen|yapılan)(.*?)(saldırılar|katliam|zulüm|soykırım|[ADJAFTER])",
            r"\s[IRK](?!istan|stan|ya)\s(tarafından)\s[IRK](?!istan|stan|ya)(a|e)(yönelik)?(.*?)(saldırılar|katliam|zulüm|soykırım|[ADJAFTER])",
            r"\s[IRK](?!istan|stan|ya)\s(tarafından)(.*?)(saldırıya|katliama|zulme|soykırıma)(.*?)(maruz kal|uğra)",
            r"\s[IRK]\s(destekli)(.*?)\s[IRK](?!istan|stan|ya)\s(darbesi|saldırıları|katliamı|soykırımı)",
            r"\s[IRK](?!istan|stan|ya)\s(.*?)(öldürdü|katletti|etnik temizlik yaptı|kirletti|bastı|şehit etti)",
            r"\s[IRK](?!istan|stan|ya)\s(tarafından)(.*?)(öldürüldü|katledildi|basıldı|şehit edildi)"
        ]
        self.misleading_nonhateful_rules = [
            r"""\s[IRK](?!istan|stan|ya)\s(çeteleri|fanatikler|fanatiği|askerleri|milisleri|militanları|yerleşimciler|milliyetçiler|güçleri|isyanı|yaygaracılığı|polisi)""",
            r"(radikal|ırkçı|fanatik)\s[IRK](?!istan|stan|ya)\s",
            r"\s[IRK](?!istan|stan|ya)(cı|ci|çi|cu|cü|çü)\sterör\sörgütü"
        ]

    def _load_external_data(self, path):
        self.target_specific_patterns = json.load(open(f"{path}/target_specific_patterns.json", "r"))
        self.hs_specific_verbs = json.load(open(f"{path}/hatespeech_indicator_verbs.json", "r"))
        self.misleading_nonhateful_patterns = json.load(open(f"{path}/misleading_nonhateful_rules.json", "r"))
        self.pre_target_keywords = json.load(open(f"{path}/pre_target_keywords.json", "r"))
        self.post_target_keywords = json.load(open(f"{path}/post_target_keywords.json", "r"))
        
        self.keywords = json.load(open(f"{path}/targeted_keywords.json", "r"))
        self.irk = "(" + "|".join(self.keywords["IRK"]) + ")"
        self.kimlik = "(" + "|".join(self.keywords["KIMLIK"]) + ")"
        self.irk_kimlik = "(" + "|".join(self.keywords["IRK"] + self.keywords["KIMLIK"]) + ")"
        
        self.pretarget_patterns = "|".join([item for sublist in list(self.pre_target_keywords.values()) for item in sublist])
        self.posttarget_patterns = "|".join([item for sublist in list(self.post_target_keywords.values()) for item in sublist])
        
        self.turkish_to_latin_map = {ord("ü"): "u", ord("ö"): "o", ord("ı"): "i", ord("ğ"): "g", ord("ş"): "s", ord("ç"): "c"}
        
    def __find_spans(self, patterns, text, key_type, degree, return_patterns):
        detected = False
        compiled_regex = re.compile(patterns)
        for r in compiled_regex.finditer(text):
            if return_patterns:
                found_text = r.group()
                first_idx, second_idx = r.span()
                if found_text[0] == " ":
                    first_idx += 1
                if found_text[-1] == " ":
                    second_idx -= 1
                found_text = found_text.strip()
                span = (first_idx, second_idx)
                self.detected_patterns[key_type].append(
                    {
                        "span": span,
                        "match": found_text,
                        "degree": degree,
                    }
                )
            detected = True
        return detected
    

    def __add_keyword_into_rule(self, rules):
        for i in range(len(rules)):
            if "[IRK_KIMLIK]" in rules[i]:
                rules[i] = rules[i].replace("[IRK_KIMLIK]", self.irk_kimlik)
            if "[IRK]" in rules[i]:
                rules[i] = rules[i].replace("[IRK]", self.irk)
            if "[KIMLIK]" in rules[i]:
                rules[i] = rules[i].replace("[KIMLIK]", self.kimlik)
            if "[ADJBEF]" in rules[i]:
                rules[i] = rules[i].replace("[ADJBEF]", self.pretarget_patterns)
            if "[ADJAFTER]" in rules[i]:
                rules[i] = rules[i].replace("[ADJAFTER]", self.posttarget_patterns)
        return rules 
    
    def _prepare_rules(self):
        self.target_agnostic_rules = self.__add_keyword_into_rule(self.target_agnostic_rules)
        self.misleading_nonhateful_rules = self.__add_keyword_into_rule(self.misleading_nonhateful_rules)
    
    @staticmethod
    def adjust_span(found_text, span):
        start, end = span
        start += found_text.startswith(" ")
        end -= found_text.endswith(" ")
        return (start, end), found_text.strip()

    def __find_spans(self, patterns, text, degree):
        compiled_regex = re.compile(patterns)
        matches = list(compiled_regex.finditer(text, re.IGNORECASE))
        detected_spans = []
        for match in matches:
            found_text = match.group()
            span, adjusted_text = self.adjust_span(found_text, match.span())
            detected_spans.append(
                {
                    "span": span,
                    "match": adjusted_text,
                    "degree": degree,
                }
            )
        return detected_spans
    
    def __detect_target_specific_patterns(self, text):
        pattern_list = []
        spans_list = []
        for degree, patterns in self.target_specific_patterns.items():
            if patterns:
                combined_patterns = r'\s({})|({})\s'.format('|'.join(patterns), '|'.join(patterns))
                found = self.__find_spans(combined_patterns, text, degree)
                spans_list.append(found)
                if len(found) > 0:
                    pattern_list.append(1)
                else:
                    pattern_list.append(0)
            else:
                pattern_list.append(0)
                spans_list.append([])
        return pattern_list, spans_list
        
    def __detect_target_agnostic_patterns(self, text):
        detected_spans = [self.__find_spans(rule, text, -1) for rule in self.target_agnostic_rules]
        filtered_spans = [spans for spans in detected_spans if len(spans) > 0]
        return len(filtered_spans) / len(self.target_agnostic_rules), filtered_spans
    
    def __detect_misleading_nonhateful_patterns(self, text):
        pattern_list = []
        span_list = []
        def has_match(rules, text, degree):
            """Helper function to check if any rule matches the text."""
            detected_spans = [self.__find_spans(rule, text, degree) for rule in rules]
            filtered_spans = [spans for spans in detected_spans if len(spans) > 0]
            return len(filtered_spans) > 0, filtered_spans
          
        for degree, patterns in self.misleading_nonhateful_patterns.items():
            if patterns:
                found_rule = False
                if degree == "20":
                    found_rule, spans = has_match(self.misleading_nonhateful_rules, text, degree)

                pattern = rf"\s({'|'.join(patterns)})|({'|'.join(patterns)})\s"
                match, match_spans = has_match([pattern], text, degree)
                if match or found_rule:
                    pattern_list.append(1)
                    span_list.append(match_spans + spans)
                else:
                    pattern_list.append(0)
                    span_list.append([])
            else:
                pattern_list.append(0)
                span_list.append([])

        return pattern_list, span_list
            
    def __detect_pre_target_patterns(self, text):
        pattern_list = []
        spans_list = []
        def matches_pattern(patterns, text, degree):
            """Helper function to check if the given patterns match the text."""
            pattern = rf"\s({'|'.join(patterns)})\s{self.irk}(?!istan|stan|ya)\s"
            # return bool(re.search(pattern, text, flags=re.I))
            detected_spans = self.__find_spans(pattern, text, degree)
            filtered_spans = [spans for spans in detected_spans if len(spans) > 0]
            return len(filtered_spans) > 0, filtered_spans

        
        for degree, patterns in self.pre_target_keywords.items():
            if patterns:
                match, spans = matches_pattern(patterns, text, degree)
                pattern_list.append(1 if match else 0)
                spans_list.append(spans)
            else:
                pattern_list.append(0)
                spans_list.append([])

        return pattern_list, spans_list

    def __detect_post_target_patterns(self, text):
        pattern_list = []
        spans_list = []

        def matches_pattern(patterns, text, degree):
            """Helper function to check if the given patterns match the text."""
            pattern = rf"\s{self.irk}(?!istan|stan|ya)\s({'|'.join(patterns)})\s"
            # return bool(re.search(pattern, text, flags=re.I))
            # return self.__find_spans(pattern, text, 'post_target', degree, return_patterns)
            detected_spans = self.__find_spans(pattern, text, degree)
            filtered_spans = [spans for spans in detected_spans if len(spans) > 0]
            return len(filtered_spans) > 0, filtered_spans
        
        for degree, patterns in self.post_target_keywords.items():
            if patterns:
                match, spans = matches_pattern(patterns, text, degree)
                pattern_list.append(1 if match else 0)
                spans_list.append(spans)
            else:
                pattern_list.append(0)
                spans_list.append([])
        return pattern_list, spans_list

    def __detect_hatespeech_indicators(self, text):
        pattern_list = []
        spans_list = []

        def matches_pattern(patterns, text, degree):
            """Helper function to check if the given patterns match the text."""
            pattern = rf"\s{self.irk}(?!istan|stan|ya)(.*?)({'|'.join(patterns)})"
            found_texts = re.findall(pattern, text, flags=re.I)
            for found_text in found_texts:
                split_text = found_text[1].split()
                if len(split_text) > 13:
                    new_text = " ".join(split_text[-13:]) + " " + found_text[2]
                    detected_spans = self.__find_spans(pattern, new_text, degree) # re.search(pattern, new_text, flags=re.I):
                    filtered_spans = [spans for spans in detected_spans if len(spans) > 0]
                    if len(filtered_spans) > 0:
                        return True, filtered_spans
                else:
                    detected_spans = self.__find_spans(pattern, text, degree) # re.search(pattern, new_text, flags=re.I):
                    filtered_spans = [spans for spans in detected_spans if len(spans) > 0]
                    return True, filtered_spans
            return False, []

        for degree, patterns in self.hs_specific_verbs.items():
            if patterns:
                match, spans = matches_pattern(patterns, text, degree)
                pattern_list.append(1 if match else 0)
                spans_list.append(spans)
            else:
                pattern_list.append(0)
                spans_list.append([])
        return (pattern_list, spans_list)
    
    @staticmethod
    def parallel_apply_wrapper(df, column, func, return_patterns):
        df[f"{column}_tuple"] = df["text"].progress_apply(lambda x: func(x))
        df[f"{column}"] = df[f"{column}_tuple"].apply(lambda x: x[0])
        df[f"{column}_spans"] = df[f"{column}_tuple"].apply(lambda x: x[1])
        if return_patterns == False:
            df.drop([f"{column}_tuple", f"{column}_spans"], axis=1, inplace=True)
        return df

    def apply_rules(self, data, return_patterns=False):
        columns_and_functions = {
            "hatespeech_indicators": self.__detect_hatespeech_indicators,
            "target_specific": self.__detect_target_specific_patterns,
            "target_agnostic": self.__detect_target_agnostic_patterns,
            "misleading_nonhateful": self.__detect_misleading_nonhateful_patterns,
            "pre_target": self.__detect_pre_target_patterns,
            "post_target": self.__detect_post_target_patterns
        }
        for column, func in columns_and_functions.items():
            data = self.__class__.parallel_apply_wrapper(data, column, func, return_patterns)
        return data

if __name__ == '__main__':
    data_path = "data/turkishprintcorpus.csv"
    data = pd.read_csv(data_path, sep=',')
    rule_assigner = LinguisticRuleGenerator()
    data =  rule_assigner.apply_rules(data, return_patterns=False)
    data.to_csv('data/turkishprintcorpus_lingfeats.csv', index=False)
