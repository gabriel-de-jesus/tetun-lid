import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from common_utils.utils import Utils
from common_utils import config
from tetuntokenizer.tokenizer import TetunSentenceTokenizer


class ProcessData:
    """
    This class performs data processing.
    """

    def __init__(self) -> None:
        self.root_txt_files = config.root_txt_files
        self.tokenizer = TetunSentenceTokenizer()
        self.punctutation_regex = config.PUNCTUATION_REGEX
        self.digit_regex = config.DIGIT_REGEX
        self.three_dots = config.THREE_DOTS
        self.hyphen_with_spaces = config.HYPHEN_WITH_SPACES

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
            sentences_list = self.tokenizer.tokenize(corpus)
            sentences_list = [s.strip() for s in sentences_list]

            # Pair langcode with each sentence
            lang_code = lang_file.split(".")[0]
            sentences_list = [(s, lang_code) for s in sentences_list if len(s) > 0]

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

    def preprocessed_data(self) -> pd.DataFrame:
        """ 
        Build a clean dataset for all the languages by removing:
        1. Punctuations.
        2. Three dots, i.e., "...".
        3. Hypens with space(s), e.g., "- " or " - ".
        and then return them in a dataframe 
        """

        data = self.save_text_data_into_dataframe()
        data.drop_duplicates(subset="sentence", keep=False, inplace=True)
        data["sentence"] = data["sentence"].str.lower()
        data["sentence"] = data["sentence"].str.replace(self.digit_regex, "")
        data["sentence"] = data["sentence"].str.replace(self.punctutation_regex, "")
        data["sentence"] = data["sentence"].str.replace(self.three_dots, "")
        data["sentence"] = data["sentence"].str.replace(self.hyphen_with_spaces, " ")

        data.reset_index(drop=True, inplace=True)
        clean_data = data[(data["sentence"] != "") & (data["sentence"] != " ")]

        return clean_data

    def initial_clean_data_with_count(self) -> pd.DataFrame:
        """ Add one new column for the sentence length to the preprocessed dataframe. """

        clean_data = self.preprocessed_data()
        clean_data["sentence_length"] = clean_data["sentence"].apply(len)

        return clean_data

    def removed_sentence_outliers(self) -> List[str]:
        """ Remove outliers (longest or shortest sentences) and return a list contains lists of 
        sentences with the respective length """

        clean_data = self.initial_clean_data_with_count()
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

    def final_clean_data(self) -> pd.DataFrame:
        """ Build a final clean dataset and return a dataframe contains sentences excluding outliers. """

        data = self.initial_clean_data_with_count()
        sentences_not_outliers = self.removed_sentence_outliers()
        # Create a dataframe with only the values that are in the sentences_not_outliers
        final_clean_dataset = data[data["sentence_length"].isin(sentences_not_outliers)]

        return final_clean_dataset
