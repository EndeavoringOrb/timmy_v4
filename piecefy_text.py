from torch.nn import functional as F
import sentencepiece as spm
from datetime import datetime as dt
import numpy as np

text_files = ["wikisent2.txt"]
save_file_name = input("Enter save name: ")
sentence_piece_model = input("Enter vocab model full relative path: ")
load_amount = 0


text = []
print("reading text")
for i in text_files:
    with open("texts/"+i, "r", encoding="utf-8") as f:
        text.append(f.readlines())

for j in range(len(text)):
    for num,i in enumerate(text[j]):
        text[j][num] = "<start>" + i[:-1] + "<end>"


# Load the trained model
print("loading model")
sp = spm.SentencePieceProcessor()
sp.load(f'{sentence_piece_model}')

pieces = []

print("characterizing text")
#pieces = sp.encode_as_pieces(text)
#print("finished")
text_len = sum([len(i) for i in text])
if load_amount != 0:
    text = text[:100_000_000]
else:
    load_amount = text_len
print_interval = text_len//10000
if print_interval == 0:
    print_interval = 1
start = dt.now()
for j in range(len(text)):
    for num, i in enumerate(text[j]):
        # Piece a sentence
        pieces.extend(sp.encode_as_ids(i))
        if num%print_interval == 0:
            end = dt.now()
            elapsed = end-start
            percent = num/text_len + 0.00000001 # messy way to avoid zero division error
            print(f"{num}/{text_len} - {percent*100:.3f}% - ETA: {(1/percent)*(elapsed)-elapsed}                     ",end="\r")
print(f"{text_len}/{text_len} - 100.00%")

print(f"length of dataset in pieces: {len(pieces)}")
print(sp.decode_ids(pieces[:1000]))

print("writing to file")
np.save(f"pieced_and_indexed_texts/{save_file_name}_pieced_{load_amount/1_000_000:.2f}M", np.array(pieces))
print("finished")