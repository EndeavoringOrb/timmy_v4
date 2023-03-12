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

def learn_bpe(text, num_merges, version=1, subwords=None, save_freq=1):
    global save_flag
    """Learns the byte pair encoding from the given text."""
    save_num = 0
    merged = []
    char_freqs = Counter()
    past_char_freq_len = -1

    print("splitting text into words")
    words = text.split()
    print(f"done. {len(words)} words")

    if subwords == None:
        # Initial merge
        pair_freq = Counter()
        print("initial pairing")
        # Define a list comprehension
        a = [
            list(zip(word[:-1], word[1:]))  # zip each word with the next one
            for word in words    # for each list of words in characterized_words
            if len(word) > 1                  # only if there are at least two words
        ]

        pair_freq = Counter([item for sublist in a for item in sublist])
        """for word in words:
            chars = list(word)
            for j in range(len(chars)-1):
                pair_freq[chars[j], chars[j+1]] += 1"""
        print("finding initial most frequent pair")
        # Find the most frequent character pair.
        most_common_pair = pair_freq.most_common(1)[0]
        print(f"most common pair: {most_common_pair}")
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
        characterized_words = new_text.split(" ")
    else:
        characterized_words = []
        for word in words:
            word = characterize(word, subwords)
            characterized_words.append(word)
        #new_text = " ".join(characterized_words)
    
    # Merge the most frequent character pairs num_merges times.
    for i in range(num_merges-1):
        new_chars = characterized_words
        #print("".join(most_common_pair) in chars)
        print("\n")
        print(f"merge num: {i+2}/{merge_iters}")
        print(f"chars: {len(new_chars)}")
        print(f"char pairs: {len(pair_freq)}")
        # Finding new frequencies
        if save_flag == True or len(char_freqs) == past_char_freq_len or i%save_freq == 0:
            with open(f"subwords_v{version}_{save_num}.txt","w") as f:
                for i in char_freqs:
                    f.write(f'{"".join(i)}\n')
            save_flag = False
            save_num += 1
        if len(char_freqs) < past_char_freq_len:
            with open(f"subwords_v{version}_{save_num}DECREASING.txt","w") as f:
                for i in char_freqs:
                    f.write(f'{"".join(i)}\n')
            save_num += 1
            print("\nVOCABULARY SIZE DECREASING - subwords saved")
        past_char_freq_len = len(char_freqs)
        print(f"length of char set: {len(char_freqs)}")

        print("Finding new frequencies")
        char_freqs = Counter([item for sublist in characterized_words for item in sublist])
        #test_freqs = Counter([1,2,3,4,5,4,3,2,1])
        characterized_words = []
        char_keys = list(char_freqs.keys())
        words_len = len(words)
        print_interval = words_len//100
        for i, word in enumerate(words):
            word = characterize(word, char_keys)
            characterized_words.append(word)
            if i%print_interval == 0:
                print(f"{i}/{words_len} - {i*100/words_len:.2f}%",end="\r")
        print(f"{words_len}/{words_len} - 100.00%")
        char_freqs = Counter([item for sublist in characterized_words for item in sublist])
        # Define a list comprehension
        a = [
            list(zip(words[:-1], words[1:]))  # zip each word with the next one
            for words in characterized_words    # for each list of words in characterized_words
            if len(words) > 1                  # only if there are at least two words
        ]

        pair_freq = Counter([item for sublist in a for item in sublist])

        print("\nFinding most common pair")
        try:
            most_common_pair = pair_freq.most_common(1)[0]
        except IndexError:
            print("No more merges. EXITING")
            break
        print(f"most common pair: {most_common_pair}")

        print("Merging most common pair")
        # Merging most common pair
        count = 0
        count2 = 0
        while count < len(characterized_words):
            while count2 < len(characterized_words[count]):
                if count < len(characterized_words)-1 and count2 < len(characterized_words[count])-1 and [characterized_words[count][count2],characterized_words[count][count2+1]] == [most_common_pair[0][0],most_common_pair[0][1]]:
                    characterized_words[count][count2] = "".join([characterized_words[count][count2],characterized_words[count][count2+1]])
                    characterized_words[count].pop(count2+1)
                count2 += 1
            count += 1
            count2 = 0
    return characterized_words, char_freqs, save_num

def save_callback(event):
    global save_flag
    save_flag = True


#######################################################################################
text_file = "wikisent2.txt"
merge_iters = 1_000_000
version = 3
load_subwords = False
subword_file = "subwords_v1_8.txt"
load_amount = 100_000 # set to 0 if you want to load everything
save_freq = 1
#######################################################################################

subwords = None

#get training text from file
print("reading file")
with open(text_file) as f:
    if load_amount <= 0:
        text = f.readlines() #64*2000
    else:
        text = f.readlines(64*load_amount)
    text = "".join(text)

if load_subwords == True:
    #get the subwords
    print("reading subwords")
    with open(subword_file) as f:
        subwords = f.readlines()
    for i in range(len(subwords)):
        subwords[i] = subwords[i].strip()

print("doing bpe thingy")
save_flag = False
keyboard.hook_key('`',save_callback)
start = dt.now()
bpe_text, char_freqs, save_num = learn_bpe(text, merge_iters, version, subwords, save_freq)
end = dt.now()
elapsed = end-start
print(f"total elapsed: {elapsed}")
print(f"avg elapsed: {elapsed/merge_iters}")
with open(f"subwords_v{version}_{save_num}END.txt","w") as f:
    for i in char_freqs:
        f.write(f'{"".join(i)}\n')
print("\n")
bpe_text_show_amount = 200
print(f"First {bpe_text_show_amount} encoded characters in {text_file}:\n")
print(bpe_text[:bpe_text_show_amount])
#print(max(pair_freqs, key=pair_freqs.get))