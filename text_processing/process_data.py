import os
import re
import numpy as np
import pandas as pd
from typing import List, Tuple
from pathlib import Path
from text_processing.dea import DataAnalysisExploration


class ProcessData:
    """
    A class for performing data processing.
    """

    def __init__(self) -> None:
        self._dea = DataAnalysisExploration()

    def load_corpus(self, file_path: Path) -> str:
        """ 
        Load and read text corpus.

        :param path: a file path.
        :return: a string text corpus.
        """
        try:
            with file_path.open('r', encoding='utf-8') as f:
                lines = f.readlines()
                text_corpus = ''.join(lines)
        except FileNotFoundError:
            print(f"File not found at: {file_path}")
            return []
        except UnicodeDecodeError:
            print(f"Cannot decode file at: {file_path}")
            return []

        return text_corpus

    def input_text_files(self, files_path: Path) -> List[str]:
        """ 
        List of input text files for all languages.

        :param files_path: a path to folder where the language files are located.
        :return: a list of text file names.
        """
        files = os.listdir(files_path)
        list_text_files = [file for file in files if file.endswith('.txt')]

        return list_text_files

    def split_to_sentences(self, files_path: Path) -> Tuple:
        """ 
        Split each language into sentences.

        :param files_path: a path to folder where the language files are located.
        :return: a tuple of lists of sentences for each language.
        """
        sentences = {
            'tet': [],
            'pt': [],
            'en': [],
            'id': []
        }

        lang_files = self.input_text_files(files_path)
        for lang_file in lang_files:
            path = Path(os.path.join(files_path, lang_file))
            corpus = self.load_corpus(path)

            # Split by delimiter .?! following by space(s)
            sentences_list = re.split(r'(?<=\w)[.?!]\s+', corpus)
            sentences_list = [s.strip() for s in sentences_list]

            # Pair langcode with each sentence
            lang_code = lang_file.split('.')[0]
            sentences_list = [(s, lang_code)
                              for s in sentences_list if len(s) > 0]

            if lang_code in sentences:
                sentences[lang_code].extend(sentences_list)

        return tuple(sentences.values())

    def compile_all_data(self, files_path: Path) -> pd.DataFrame:
        """ 
        Save dataset for all four languages in a data frame.

        :param files_path: a path to folder where the language files are located.
        :return: a data frame contains sentences with the respective language.
        """
        tet_sentences, pt_sentences, en_sentences, id_sentences = self.split_to_sentences(
            files_path)
        all_data = tet_sentences + pt_sentences + en_sentences + id_sentences
        dataset = pd.DataFrame(all_data, columns=['sentence', 'language'])
        dataset.reset_index(drop=True, inplace=True)

        return dataset

    def preprocessed_data(self, files_path: Path) -> pd.DataFrame:
        """ 
        Build a clean dataset for all four languages and save in a data frame.

        :param files_path: a path to folder where the language files are located.
        :retun: a data frame contains sentences with the respective language.
        """
        punctuation = '!\"“”#$€&()*+,./–:;<=>?@%[\\]^_`{|}~\n'
        punctutation_regex = r"[" + re.escape("".join(punctuation)) + "]"
        digit_regex = r"\d+"
        three_dots = r"[…]+"
        hyphen_with_spaces = r"\s*-\s+"

        data = self.compile_all_data(files_path)
        data.drop_duplicates(subset='sentence', keep=False, inplace=True)
        data['sentence'] = data['sentence'].str.lower()
        data['sentence'] = data['sentence'].str.replace(digit_regex, "")
        data['sentence'] = data['sentence'].str.replace(punctutation_regex, "")
        data['sentence'] = data['sentence'].str.replace(three_dots, "")
        data['sentence'] = data['sentence'].str.replace(
            hyphen_with_spaces, " ")

        data.reset_index(drop=True, inplace=True)
        clean_data = data[(data['sentence'] != '') & (data['sentence'] != ' ')]

        return clean_data

    def words_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """ 
        Summarize the words in each sentence and the overall total.

        :param data: a DataFrame contained the preprocessed data.
        :return: a data frame contains the summary of words for each sentence.
        """
        summary = []
        for lang in list(set(data['language'])):
            sentence_list = data['sentence'][data['language'] == lang]
            words = sentence_list.str.split()
            words_count = words.apply(len)
            max_words = words_count.max()
            min_words = words_count.min()
            avg_words = words_count.mean()
            total_words = words_count.sum()

            summary.append({
                'language': lang,
                'max_words/sentence': max_words,
                'min_words/sentence': min_words,
                'avg_words/sentence': avg_words,
                'total_words_in_doc': total_words,
            })

        return pd.DataFrame(summary)

    def clean_data_with_count(self, files_path: Path) -> pd.DataFrame:
        """ 
        Build a clean dataset with a new column contains sentence length.

        :param files_path: a path to folder where the language files are located.
        :return: a data frame contains sentences including its length.
        """
        clean_data = self.preprocessed_data(files_path)
        clean_data['sentence_length'] = clean_data['sentence'].apply(
            len)

        return clean_data

    def removed_sentence_outliers(self, data: pd.DataFrame) -> List[str]:
        """
        Remove outliers (longest or shortest sentences) from the data

        :param data: a DataFrame contained the preprocessed data.
        :return: a list contains list of Tetun (tet), Portuguese (pt), English (en), 
            and Indonesian (id) language
        """
        data_counts = self._dea.count_sentences(data)

        for i in range(len(data_counts)):
            cut_off_1 = np.std(data_counts[i]) * 1
            mean_value = np.mean(data_counts[i])
            lower, upper = mean_value - cut_off_1, mean_value + cut_off_1

            outlier_removed = [[el for el in sublist if el >
                                lower and el < upper] for sublist in data_counts]

        return outlier_removed

    def final_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """ 
        Build a final clean dataset excluding the outliers.

        :param data: a DataFrame contained the preprocessed data.
        :return: a data frame contains sentences excluding its length.
        """
        sentences_not_outliers = self.removed_sentence_outliers(data)

        # Extract sentence_length from the sentences_not_outliers
        values_to_keep = []
        for sentence_not_outlier in sentences_not_outliers:
            for per_language_value in sentence_not_outlier:
                values_to_keep.append(per_language_value)

        # Create a data frame with only the values that are in the sentences_not_outliers
        final_clean_data = data[data['sentence_length'].isin(values_to_keep)]
        # Drop the sentence_length column
        final_clean_dataset = final_clean_data.drop('sentence_length', axis=1)

        return final_clean_dataset
