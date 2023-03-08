import os
import re
import numpy as np
import pandas as pd
from text_processing.dea import count_sentences


def load_corpus(file_path: str) -> str:
    """ Load and read text corpus.

    Args:
      path (str): a file path.

    Returns:
      A string text corpus.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        text_corpus = ''.join(lines)

    return text_corpus


def input_text_files(files_path: str) -> list:
    """ List of input text files for all languages.

    Args:
      files_path (str): a path to folder where the language files are located.

    Returns:
        A list of text file names.
    """
    files = os.listdir(files_path)
    list_text_files = [file for file in files if file.endswith('.txt')]

    return list_text_files


def split_to_sentences(files_path: str) -> tuple:
    """ Split each language into sentences.

    Args:
        files_path (str): A path to folder where the language files are located.

    Returns:
        A tuple of lists of sentences for each language.
    """
    sentences = {
        'tet': [],
        'pt': [],
        'en': [],
        'id': []
    }

    lang_files = input_text_files(files_path)
    for lang_file in lang_files:
        corpus = load_corpus(os.path.join(files_path, lang_file))

        # Split by delimiter .?! following by space(s)
        sentences_list = re.split(r'(?<=\w)[.?!]\s+', corpus)
        sentences_list = [s.strip() for s in sentences_list]

        # Pair langcode with each sentence
        lang_code = lang_file.split('.')[0]
        sentences_list = [(s, lang_code) for s in sentences_list if len(s) > 0]

        if lang_code in sentences:
            sentences[lang_code].extend(sentences_list)

    return tuple(sentences.values())


def compile_all_data(files_path: str) -> pd.DataFrame:
    """ Save dataset for all four languages in a data frame.

      Args:
          files_path (str): A path to folder where the language files are located.

      Returns:
          A data frame contains sentences with the respective language.
      """
    tet, pt, en, id = split_to_sentences(files_path)
    all_data = tet + pt + en + id
    dataset = pd.DataFrame(all_data, columns=['sentence', 'language'])
    dataset.reset_index(drop=True, inplace=True)

    return dataset


def preprocessed_data(files_path: str) -> pd.DataFrame:
    """ Build a clean dataset for all four languages and save in a data frame.

      Args:
          files_path (str): A path to folder where the language files are located.

      Returns:
          A data frame contains sentences with the respective language.
      """
    punctuation = '!\"“”#$€&()*+,./–:;<=>?@%[\\]^_`{|}~'
    punctutation_regex = r"[" + re.escape("".join(punctuation)) + "]"
    digit_regex = r"\d+"
    three_dots = r"[…]+"
    hyphen_with_spaces = r"\s*-\s+"

    data = compile_all_data(files_path)
    data.drop_duplicates(subset='sentence', keep=False, inplace=True)
    data['sentence'] = data['sentence'].str.lower()
    data['sentence'] = data['sentence'].str.replace(digit_regex, "")
    data['sentence'] = data['sentence'].str.replace(punctutation_regex, "")
    data['sentence'] = data['sentence'].str.replace(three_dots, "")
    data['sentence'] = data['sentence'].str.replace(hyphen_with_spaces, " ")

    data.reset_index(drop=True, inplace=True)

    clean_data = data[data['sentence'] != '']

    return clean_data


def clean_data_with_count(files_path: str) -> pd.DataFrame:
    """ Build a clean dataset with a new column contains sentence length.

      Args:
          files_path (str): A path to folder where the language files are located.

      Returns:
          A data frame contains sentences including its length.
      """
    clean_data = preprocessed_data(files_path)
    clean_data['sentence_length'] = clean_data['sentence'].apply(
        len)

    return clean_data


def removed_sentence_outliers(data: pd.DataFrame) -> list:
    """ Remove outliers (longest or shortest sentences) from the data

    Args:
        data (DataFrame): A DataFrame contained the preprocessed data.

    Returns:
        A list contains list of Tetun (tet), Portuguese (pt), English (en), 
        and Indonesian (id) language
    """
    data_counts = count_sentences(data)

    for i in range(len(data_counts)):
        cut_off_1 = np.std(data_counts[i]) * 1
        mean_value = np.mean(data_counts[i])
        lower, upper = mean_value - cut_off_1, mean_value + cut_off_1

        outlier_removed = [[el for el in sublist if el >
                            lower and el < upper] for sublist in data_counts]

    return outlier_removed


def final_clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """ Build a final clean dataset excluding the outliers.

      Args:
          data (DataFrame): A DataFrame contained the preprocessed data.

      Returns:
          A data frame contains sentences excluding its length.
      """
    sentences_not_outliers = removed_sentence_outliers(data)

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
