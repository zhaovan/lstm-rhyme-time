import numpy as np
import csv
import string
import pronouncing


def get_data(file_path):
    # set of characters to exclude
    exclude = set(['!', '.', ',', '?', "'", '"', "#"])
    # set of phonemes to exclude, since the CMU dict sometimes returns these words instead of phoenems
    phoneme_exclude = set(["#", "foreign", "french", "abbrev"])
    # set of special_chars that do NOT get converted to phonemes
    special_chars = set(["NEWLINE", "SPACE", "END"])
    # open csv file
    with open(file_path) as f:
        csv_reader = csv.reader(f)
        data = []
        for row in csv_reader:
            # get rid of characters from the exclude set
            row[0] = "".join(c for c in row[0] if c not in exclude)
            # put in appropriate SPACE and NEWLINE tokens
            row[0] = row[0].replace(" ", " SPACE ")
            row[0] = (row[0].replace("\n", " NEWLINE "))
            row[0] = row[0].split(" ")
            poem = []
            # strip special symbols
            for w in row[0]:
                w = w.strip("'")
                w = w.strip('"')
                w = w.strip("(")
                w = w.strip(")")
                if w != "":
                    poem.append(w)
            # put an END token at the end of a poem
            poem[-1] = "END"
            data.append(poem)

        phonemes_data = []
        for poem in data:
            poem_phonemes = []
            # for each word in a poem, convert it to it's phoneme representation, unless it's a SPACE, NEWLINE, or END token
            for word in poem:
                if word not in special_chars:
                    p_list = pronouncing.phones_for_word(word)
                    phoneme_word = ""
                    if (len(p_list) > 0):
                        phoneme_word = p_list[0]
                        phoneme_list = phoneme_word.split(" ")
                    else:
                        phoneme_list = ["UNK" for i in range(int(len(word)/ 2))]
                    final_phoneme_list = [p for p in phoneme_list if p not in phoneme_exclude]
                    poem_phonemes += final_phoneme_list
                else:
                    poem_phonemes.append(word)
            poem_len = len(poem_phonemes)
            # only include poems with phoneme length 120-150, so that the structure of our data remain consistent
            if poem_len <= 150 and poem_len >= 120:
                phonemes_data.append([poem_phonemes, poem_len])
        padded_phonemes_data = []

        # pad poems with PAD tokens so that they're all the same length (cannot feed into LSTM otherwise)
        for (poem, poem_len) in phonemes_data:
            padded_poem = poem 
            padded_poem += ["PAD" for _ in range(150 - poem_len)]
            padded_phonemes_data.append(padded_poem)

        # set of unique phonemes
        unique_phonemes = set([phoneme for poem in padded_phonemes_data for phoneme in poem])
        
        phonemes_dict = {}
        i = 0

        # create phoneme dict with unique ids for every phoneme (including special tokens END, SPACE, NEWLINE, and PAD)
        for s in unique_phonemes:
            phonemes_dict[s] = i
            i += 1
        phoneme_indexes = [[phonemes_dict[s] for s in poem] for poem in padded_phonemes_data]

        # return the phoneme indexes, the dictionary, the length of the dictionary, and the unique ID of PAD
        return phoneme_indexes, phonemes_dict, len(phonemes_dict), phonemes_dict["PAD"]

if __name__ == "__main__":
    get_data("data/limericks.csv")