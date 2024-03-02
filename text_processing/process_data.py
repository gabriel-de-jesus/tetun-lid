import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from common_utils.utils import Utils
from common_utils import config
from tetuntokenizer.tokenizer import TetunSentenceTokenizer, TetunWordTokenizer

#!/usr/bin/env python
#
# process_data.py
# Gabriel de Jesus (mestregabrieldejesus@gmail.com)
# 05-04-2023


class ProcessData:
    """
    This class:
    1. Load each of the language files from the lang_files folder.
    2. Split into tokens using Tetun sentence tokenizer and map each sentence to the respective langID.
    3. Store sentence - langID pairs in dataframe.
    4. Preprocess sentences by removing punctuations, symbols, digits, etc.
    """

    def __init__(self) -> None:
        self.root_txt_files = config.root_txt_files
        self.sentence_tokenizer = TetunSentenceTokenizer()
        self.word_tokenizer = TetunWordTokenizer()
        self.punctutations_symbols_regex = config.PUNCTUATIONS_SYMBOLS_REGEX
        self.digits_regex = config.DIGITS_REGEX
        self.int_numbers_regex = config.INT_NUMBERS_REGEX
        self.three_dots = config.THREE_DOTS
        self.hyphen_with_spaces = config.HYPHEN_WITH_SPACES
        self.one_or_more_spaces = config.ONE_OR_MORE_SPACES
        self.remove_space_at_the_beginning = config.REMOVE_SPACE_AT_THE_BEGINNING

    def list_of_input_text_files(self) -> List[str]:
        """ Get lists of the input text files and return a list of their corresponding file names. """

        files = os.listdir(self.root_txt_files)
        list_text_files = [file for file in files if file.endswith(".txt")]

        return list_text_files

    def split_text_into_sentences(self) -> Tuple:
        """ Split each language in the input text into sentences and 
        return a tuple of sentence and language pairs for each language. """

        sentences = {"tet": [], "pt": [], "en": [], "id": []}

        lang_files = self.list_of_input_text_files()
        for lang_file in lang_files:
            path = Path(os.path.join(self.root_txt_files, lang_file))
            corpus = Utils(path).load_corpus()

            # tokenize by sentence
            sentences_list = self.sentence_tokenizer.tokenize(corpus)
            sentences_list = [s.strip() for s in sentences_list]

            # Pair langcode with each sentence
            lang_code = lang_file.split(".")[0]
            sentences_list = [(s, lang_code)
                              for s in sentences_list if len(s) > 0]

            if lang_code in sentences:
                sentences[lang_code].extend(sentences_list)

        return tuple(sentences.values())

    def save_text_data_into_dataframe(self) -> pd.DataFrame:
        """ Save dataset for all the languages in a dataframe and 
        return a dataframe contains sentences with the respective language. """

        tet_sentences, pt_sentences, en_sentences, id_sentences = self.split_text_into_sentences()
        all_data = tet_sentences + pt_sentences + en_sentences + id_sentences
        dataset = pd.DataFrame(all_data, columns=["sentence", "language"])
        dataset.reset_index(drop=True, inplace=True)

        return dataset

    def preprocess_dataset(self) -> pd.DataFrame:
        """ 
        Build a clean dataset for all the languages by removing:
        1. Digits.
        2. Punctuations.
        3. Three dots, i.e., "...".
        4. Hypens with space(s), e.g., "- " or " - ".
        5. Space at the beginning of the sentence.
        6. Integer values resulting from (2), i.e., in (1) digits like 12/01 are not removed, and in (2) it becomes 12 02.
        7. Replace one or more consecutive spaces with one only space.
        and then return them in a dataframe 
        """

        data = self.save_text_data_into_dataframe()
        data.drop_duplicates(subset="sentence", keep=False, inplace=True)

        data['sentence'] = data['sentence'].str.lower()
        # E.g. 12.000.000,05 or 12,000,000.05.
        data['sentence'] = data['sentence'].apply(
            lambda x: re.sub(self.digits_regex, " ", x))
        # For the numbers, e.g., 12/03 becomes 12 03.
        data['sentence'] = data['sentence'].apply(
            lambda x: re.sub(self.punctutations_symbols_regex, " ", x))
        data['sentence'] = data['sentence'].apply(
            lambda x: re.sub(self.three_dots, "", x))
        data['sentence'] = data['sentence'].apply(
            lambda x: re.sub(self.hyphen_with_spaces, " ", x))
        data['sentence'] = data['sentence'].apply(
            lambda x: re.sub(self.remove_space_at_the_beginning, r'\1', x))
        # Remove integer digits, e.g., 12 03.
        data['sentence'] = data['sentence'].apply(
            lambda x: re.sub(self.int_numbers_regex, " ", x))
        data['sentence'] = data['sentence'].apply(
            lambda x: re.sub(self.one_or_more_spaces, " ", x))
        data.loc[data['sentence'].str.len() < 10, 'sentence'] = ""

        data.reset_index(drop=True, inplace=True)
        clean_data = data[(data["sentence"] != "") & (data["sentence"] != " ")]
        return clean_data

    def preprocess_dataset_with_count(self) -> pd.DataFrame:
        """ Add one new column for the sentence length to the preprocessed dataframe. """

        clean_data = self.preprocess_dataset()
        clean_data["sentence_length"] = clean_data["sentence"].apply(len)

        return clean_data

    def remove_outliers_data(self) -> List[str]:
        """ Remove outliers (longest or shortest sentences) and return a list contains lists of 
        sentences with the respective length """

        clean_data = self.preprocess_dataset_with_count()
        data_counts = clean_data["sentence_length"]
        outlier_removed = []

        for i, row in clean_data.iterrows():
            length = row["sentence_length"]
            # sentence = row["sentence"]
            cut_off_1 = np.std(data_counts) * 1
            mean_value = np.mean(data_counts)
            lower, upper = mean_value - cut_off_1, mean_value + cut_off_1

            if length > lower and length < upper:
                outlier_removed.append(length)

        return outlier_removed

    def normalize_dataset(self) -> pd.DataFrame:
        """ Tokenize each sentence into individual words, pairing each word with its respective language. """

        data = self.preprocess_dataset_with_count()
        sentences_not_outliers = self.remove_outliers_data()
        # Create a dataframe with only the values that are in the normalize_sentences
        final_clean_dataset = data[data["sentence_length"].isin(
            sentences_not_outliers)]

        return final_clean_dataset

    def tokenize_sentence_to_words(self) -> pd.DataFrame:
        """ Tokenize each sentence into individual words with the respective language pairs. """
        tokenized_data = []
        for _, row in self.normalize_dataset().iterrows():
            tokens = self.word_tokenizer.tokenize(row["sentence"])
            for token in tokens:
                tokenized_data.append({'token': token, 'language': row["language"]})

        return pd.DataFrame(tokenized_data)
