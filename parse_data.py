from common_utils.utils import Utils
from common_utils import config


class ParseData:
    def __init__(self) -> None:
        self.tet_csv_file_path = config.tet_csv
        self.pt_lzma_file_path = config.pt_lzma
        self.en_lzma_file_path = config.en_lzma
        self.id_lzma_file_path = config.id_lzma
        self.tet_text_file_path = config.tetun
        self.pt_text_file_path = config.portuguese
        self.en_text_file_path = config.english
        self.id_text_file_path = config.indonesian
        self.bite_size = config.BITE_SIZE

    def run_tetun_parse(self) -> None:
        """ Parse Tetun dataset from csv file to text file if it does not parse yet. """

        if not self.tet_text_file_path:
            tetun = Utils(self.tet_csv_file_path, self.tet_text_file_path)
            tetun.parse_tetun_corpus()
        else:
            print(f"Tetun csv file is already parsed.")

    def run_other_lang_parse(self) -> None:
        """ Parse pt, en, and id datasets from xz files to text files if they do not parse yet. """

        if not self.pt_text_file_path:
            portuguese = Utils(self.pt_lzma_file_path, self.pt_text_file_path)
            portuguese.parse_corpus_of_other_lang()
        else:
            print(f"Portuguese lzma file is already parsed.")

        if not self.en_text_file_path:
            english = Utils(self.en_lzma_file_path, self.en_text_file_path)
            english.parse_corpus_of_other_lang()
        else:
            print(f"English lzma file is already parsed.")

        if not self.id_text_file_path:
            indonesian = Utils(self.id_lzma_file_path, self.id_text_file_path)
            indonesian.parse_corpus_of_other_lang()
        else:
            print(f"Indonesian lzma file is already parsed.")


if __name__ == "__main__":
    parse_data = ParseData()
    parse_data.run_tetun_parse()
    parse_data.run_other_lang_parse()
