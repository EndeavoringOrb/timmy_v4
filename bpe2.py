from collections import defaultdict, Counter
from datetime import datetime as dt
import keyboard

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


#characterized = characterize("hello",["hel","o","ll","el","lo","hel"])
#print(characterized)
#input("waiting thing")

def learn_bpe(text, num_merges, version=1, subwords=None):
    global save_flag
    """Learns the byte pair encoding from the given text."""
    save_num = 0
    merged = []
    char_set = []
    past_char_set_len = -1

    print("splitting text into words")
    words = text.split()
    print(f"done. {len(words)} words")

    if subwords == None:
        # Initial merge
        for i in range(1):
            pair_freq = Counter()
            print("initial pairing")
            for word in words:
                chars = list(word)
                for j in range(len(chars)-1):
                    pair_freq[chars[j], chars[j+1]] += 1
            if not pair_freq:
                break
            print("finding initial most frequent pair")
            # Find the most frequent character pair.
            most_common_pair = max(pair_freq, key=pair_freq.get)
            merged.append(most_common_pair)
            freq = pair_freq[most_common_pair]
            new_key = "".join(most_common_pair)
            pair_freq[new_key] = freq
            del pair_freq[most_common_pair]
            print("merge inital most frequent pair")
            # Merge the most frequent character pair.
            new_text = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and (word[i], word[i+1]) == most_common_pair:
                        new_word.append(word[i]+word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_text.append(" ".join(new_word))
            new_text = " ".join(new_text)
    else:
        characterized_words = []
        print(len(words))
        for word in words:
            word = characterize(word, subwords)
            characterized_words.extend(word)
        #new_text = " ".join(characterized_words)
    
    # Merge the most frequent character pairs num_merges times.
    for i in range(num_merges-1):
        if i == 0 and subwords != None:
            new_chars = characterized_words
        else:
            new_chars = new_text.split(" ")
        char_set = set(char_set+new_chars)
        #print("".join(most_common_pair) in chars)
        print("\n")
        print(f"merge num: {i+2}/{merge_iters}")
        print(f"chars: {len(new_chars)}")
        print(f"char pairs: {len(pair_freq)}")
        # Finding new frequencies
        pair_freq = Counter()
        words_len = len(words)
        #char_set = set(chars)
        if save_flag == True or len(char_set) == past_char_set_len:
            with open(f"subwords_v{version}_{save_num}.txt","w") as f:
                for i in char_set:
                    f.write(f'{"".join(i)}\n')
            save_flag = False
            save_num += 1
        if len(char_set) < past_char_set_len:
            with open(f"subwords_v{version}_{save_num}DECREASING.txt","w") as f:
                for i in char_set:
                    f.write(f'{"".join(i)}\n')
            save_num += 1
            print("\nVOCABULARY SIZE DECREASING - subwords saved")
        past_char_set_len = len(char_set)
        print(f"length of char set: {len(char_set)}")

        print("Finding new frequencies")
        for num, word in enumerate(words):
            print(f"{num+1}/{words_len} - {(num+1)*100/words_len:.2f}%",end="\r")
            word_chars = characterize(word, char_set)
            if len(word_chars) == 1:
                continue
            for j in range(len(word_chars)-1):
                pair_freq[word_chars[j], word_chars[j+1]] += 1

        print("\nFinding most common pair")
        # Find the most common character pair.
        try:
            most_common_pair = max(pair_freq, key=pair_freq.get)
            merged.append(most_common_pair)
        except ValueError:
            break
        print(f"most common pair: {most_common_pair}")

        print("Merging most common pair")
        # Merging most common pair
        new_word = []
        new_text = []
        for word in words:
            word = characterize(word, char_set)
            word_len = len(word)
            new_word = [word[i] + word[i+1] if i < word_len-1 and [word[i], word[i+1]] == [most_common_pair[0],most_common_pair[1]] else word[i] for i in range(word_len)]
            new_text.append(' '.join(new_word))
        new_text = " ".join(new_text)
    return new_text, char_set, save_num

def save_callback(event):
    global save_flag
    save_flag = True


#######################################################################################
text_file = "wikisent2.txt"
merge_iters = 2
version = 2
load_subwords = False
subword_file = "subwords_v1_8.txt"
#######################################################################################

subwords = None

#get training text from file
print("reading file")
with open(text_file) as f:
    text = f.readlines(64*200) #64*2000
    text = "".join(text)

if load_subwords == True:
    #get the subwords
    print("reading subwords")
    with open(subword_file) as f:
        subwords = f.readlines()
    for i in range(len(subwords)):
        subwords[i] = subwords[i].strip()

print("doing bpe thingy")
start = dt.now()
save_flag = False
keyboard.hook_key('`',save_callback)
bpe_text, char_set, save_num = learn_bpe(text, merge_iters, version, subwords)
end = dt.now()
elapsed = end-start
print(f"total elapsed: {elapsed}")
print(f"avg elapsed: {elapsed/merge_iters}")
with open(f"subwords_v{version}_{save_num}END.txt","w") as f:
    for i in char_set:
        f.write(f'{"".join(i)}\n')
print("\n")
bpe_text_show_amount = 1000
print(f"First {bpe_text_show_amount} encoded characters in {text_file}:\n")
print(bpe_text[:bpe_text_show_amount])
#print(max(pair_freqs, key=pair_freqs.get))