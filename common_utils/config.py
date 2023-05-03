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

# File size
BITE_SIZE = 3834110
DATA_FRAC_FOR_CLUSTERING = 0.03

# Punctuations
PUNCTUATION = '!"“”#$€&()*+,./–:;<=>?@%[\\]^_`{|}~\n'
PUNCTUATION_REGEX = r"[" + re.escape("".join(PUNCTUATION)) + "]"
DIGIT_REGEX = r"\d+"
THREE_DOTS = r"[…]+"
HYPHEN_WITH_SPACES = r"\s*-\s+"
