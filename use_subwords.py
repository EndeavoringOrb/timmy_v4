

text_file = "wikisent2.txt"
subword_file = "subwords_v1_11.txt"

def characterize(word, characters):
    break_flag = False
    new_word = list(word)
    for character in characters:
        n = len(character)
        for i in range(len(word) - n + 1):
            for j in range(n,-1,-1):
                if ''.join(new_word[i:i+n-j]) == character:
                    new_word[i:i+n-j] = [character]
                    break_flag = True
                    break
            if break_flag == True:
                break_flag = False
                break
    return new_word

#get training text from file
print("reading text")
with open(text_file) as f:
    text = f.readlines(64*20)
    text = "".join(text)

#get the subwords
print("reading subwords")
with open(subword_file) as f:
    subwords = f.readlines()
for i in range(len(subwords)):
    subwords[i] = subwords[i].strip()
    
subwords = sorted(subwords, key=len, reverse=True)

words = text.split()
characterized_words = []
for word in words[:1000]:
    word = characterize(word, subwords)
    characterized_words.extend(word)

print(" ".join(characterized_words))