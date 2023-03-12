import torch
import sentencepiece as spm
import torch.nn as nn
from torch.nn import functional as F
from train_funcs2 import BigramLanguageModel, Block, MultiHeadAttention, Head, BatchNorm1d, FeedForward
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sentence_piece_model = input("Enter vocab model: ")
model_file_name = input("Enter the path of the ai model you want to use: ")
generation_length = int(input("How long do you want the generated text to be? "))
model_context = input("Model Prompt (press enter if you dont want to prompt): ")
vocab_size = 8192

def encode(pieces):
    return sp.encode_as_ids(pieces)
def decode(pieces):
    return sp.decode_ids(pieces)

# Load the trained model
model = torch.load(model_file_name)

print("loading model")
sp = spm.SentencePieceProcessor()
sp.load(f'{sentence_piece_model}')

print("encoding prompt")
if model_context != "":
    print(sp.encode_as_pieces(model_context))
    model_context = np.array(encode(model_context))
    model_context = np.reshape(model_context,(model_context.shape[0],1))
    print(model_context)
    context = torch.tensor(np.array(model_context), dtype=torch.long)
else:
    context = torch.zeros((1,1), dtype=torch.long, device=device)

print("generating")

new_text = decode(model.generate(context, max_new_tokens=generation_length)[0].tolist())
#print(new_text[:1000])
generated_num = len(os.listdir("generated"))
with open(f"generated/generated{generated_num}.txt", "w", encoding="utf-8") as f:
    f.write(new_text)