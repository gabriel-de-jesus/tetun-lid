import pandas as pd
import lzma
from pathlib import Path
from bs4 import BeautifulSoup as bs


class Utils:
    """ This class contains functions for parsing and reading files """

    def __init__(self, origin_file_path: Path, target_file_path: Path = None) -> None:
        self.origin_file_path = origin_file_path
        self.target_file_path = target_file_path

    def parse_tetun_corpus(self) -> None:
        """ Load Tetun dataset from csv file, clean the HTML tags, and save them in a text file. """

        try:
            tetun_corpus = pd.read_csv(self.origin_file_path)
        except FileNotFoundError as e:
            print(f"File not found at: {e}")
            return []

        clean_corpus = [
            bs(content, "html.parser").text.strip() for content in tetun_corpus["content"]
        ]

        for clean_doc in clean_corpus:
            with self.target_file_path.open("a", encoding="utf-8") as clean_corpus_f:
                clean_corpus_f.write(clean_doc + "\n\n")

    def parse_corpus_of_other_lang(self, byte_size: int) -> None:
        """ Load EN, PT and ID compressed files in XZ format and save them in the text files. """

        try:
            with lzma.open(self.origin_file_path, "rb") as f:
                # The size is set in accordance with the size of the Tetun text
                contents = f.read(byte_size).decode("utf-8")

        except FileNotFoundError as e:
            print(f"File not found at: {e}")
            return []

        except lzma.LZMAError as e:
            print(f"Error: failed to decompress lzma file: {e}.")
            return []

        except UnicodeDecodeError as e:
            print(f"Error: failed to decode content from file: {e}")
            return []

        with self.target_file_path.open("w") as contents_f:
            contents_f.write(contents)

    def load_corpus(self) -> str:
        """ Load and read a text corpus. """

        try:
            with self.origin_file_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                text_corpus = "".join(lines)
        except FileNotFoundError:
            print(f"File not found at: {self.origin_file_path}")
            return []
        except UnicodeDecodeError:
            print(f"Cannot decode file at: {self.origin_file_path}")
            return []

        return text_corpus
