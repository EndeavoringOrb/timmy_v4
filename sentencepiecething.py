import sentencepiece as spm
import os

main_dir = 'texts'
text_files = [main_dir+"/"+f for f in os.listdir(main_dir) if os.path.isfile(os.path.join(main_dir, f))]

spm.SentencePieceTrainer.train(input=text_files, model_prefix='models_and_vocabs/harry_potter_and_GOT', vocab_size=8192*2, user_defined_symbols=['<start>', '<end>'], train_extremely_large_corpus=1, input_sentence_size=1_000_000)
# 50304