import re
from pathlib import Path

# File paths
tet_csv = Path("lang_files_origin/tetun.csv")
pt_lzma = Path("lang_files_origin/pt.txt.xz")
en_lzma = Path("lang_files_origin/en.txt.xz")
id_lzma = Path("lang_files_origin/id.txt.xz")
tetun = Path("lang_files/tet.txt")
portuguese = Path("lang_files/pt.txt")
english = Path("lang_files/en.txt")
indonesian = Path("lang_files/id.txt")
root_txt_files = Path("lang_files")
plots = Path("plots/")

# Customize file size
BITE_SIZE = 3834110
DATA_FRAC_FOR_CLUSTERING = 0.1

# Regular expressions (regex)
# E.g., 12.000.678,05 or 12,000,000.05.
DIGITS_REGEX = r"\s+[\d]+(?:[\.\,][\d]*)?\s+"
PUNCTUATION_SYMBOLS = '!"“”(),./:;?[\\]^_`{|}#&§©®™°∞¶†‡$€£μ@*+÷%<=>«»~\n'
PUNCTUATIONS_SYMBOLS_REGEX = r"[" + \
    re.escape("".join(PUNCTUATION_SYMBOLS)) + "]"
THREE_DOTS = r"[…]+"
REMOVE_SPACE_AT_THE_BEGINNING = r"^\s*([a-z])"  # Will be removed using r'\1'.
HYPHEN_WITH_SPACES = r"\s*–*-*\s+"  # E.g., space-space, word-space.
# E.g. 12, 150, 2000, etc, excluding na'in-56 or ema-2.
INT_NUMBERS_REGEX = r"\b(?<!-)\d+\b"
# Eg. space space -> will be replace with the one space.
ONE_OR_MORE_SPACES = r"\s+\s*"
