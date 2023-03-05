import os
import lzma
import pandas as pd
from bs4 import BeautifulSoup as bs


def load_tetun_csv_data(csv_file_name, target_file):
    """ Load Tetun dataset from csv file, clean the HTML tags, & save them to text file.

    Args:
        csv_file_name (str): a file in csv format.
        target_file (str): a string text file name (or path) to save the output text.
    """
    tetun_corpus = pd.read_csv(csv_file_name)
    clean_corpus = []
    for content in tetun_corpus["content"]:
        parser_content = bs(content, "html.parser").text.strip()
        clean_corpus.append(parser_content)

    for clean in clean_corpus:
        with open(target_file, "a", encoding="utf-8") as file:
            # Use two newlines to separate each document
            file.write(clean + '\n\n')


def load_other_languages_corpus(file_name, target_file):
    """ Load EN, PT and ID compressed files in XZ format and save them to text file.

    Args:
        file_name (str): a list of file names.
        target_file (str): a string text file name (or path) to save the output text.
    """
    with lzma.open(file_name, 'rb') as f:
        # The size is set in accordance with the size of Tetun text
        read_content = f.read(3834110)
        content_decoded = read_content.decode('utf-8')

    with open(target_file, 'w') as f:
        f.write(content_decoded)


if __name__ == '__main__':

    current_dir = os.getcwd()
    lang_dir = "lang_files/"
    data_dir = '/'.join([current_dir, lang_dir])

    # Tetun
    tetun_csv_file = "tn-tetun.csv"
    tetun_input_file = "/".join([data_dir, tetun_csv_file])
    tetun_text_file = "tet.txt"
    tetun_output_file = "/".join([data_dir, tetun_text_file])

    # Generate Tetun dataset if it does not existed
    if not tetun_text_file in os.listdir(lang_dir):
        load_tetun_csv_data(tetun_input_file, tetun_output_file)
    else:
        print(f"The {tetun_text_file} is already generated.")

    # Generate other languages
    file_names = ['pt.txt.xz', 'id.txt.xz', 'en.txt.xz']

    for file_name in file_names:
        if file_name in os.listdir(lang_dir):
            load_other_languages_corpus(data_dir + file_name,
                                        data_dir + file_name[:2] + ".txt")
    else:
        print(
            "The XZ files do not exist in the folder of lang_files")
