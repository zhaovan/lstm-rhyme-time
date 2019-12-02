import numpy as np
import csv
import string
import pronouncing


def get_data(file_path):
    # add any characters to this set if you don't want it in the final data
    exclude = set(['!', '.', ',', '?', "'", '"'])
    special_chars = set(["NEWLINE", "SPACE", "END"])
    # def flatten(l): return [item for sublist in l for item in sublist]
    unique_syllables = set([])
    with open(file_path) as f:
        csv_reader = csv.reader(f)
        data = []
        for row in csv_reader:
            row[0] = "".join(c for c in row[0] if c not in exclude)

            row[0] = row[0].replace(" ", " SPACE ")
            row[0] = (row[0].replace("\n", " NEWLINE "))
            row[0] = row[0].split(" ")
            poem = []
            for w in row[0]:
                w = w.strip("'")
                w = w.strip('"')
                w = w.strip("(")
                w = w.strip(")")
                if w != "":
                    poem.append(w)
            poem[-1] = "END"
            data += poem
        print(data[:200])
        syllables = []
        for w in data:
            if w not in special_chars:
                p_list = pronouncing.phones_for_word(w)
                # if syllables exist in dict, get the first one
                syllable_word = ""
                if(len(p_list) > 0):
                    syllable_word = p_list[0]
                    syllable_list = syllable_word.split(" ")
                # else, make an UNK syllable list of length
                # where length is word length divided by 3
                # [:-1] removes last space in UNK list
                else:
                    syllable_list = ["UNK" for i in range(int(len(w) / 3))]
                    # syllable_word = ("UNK " * int(len(w) / 3))[:-1]
                syllables += syllable_list
            else:
                syllables.append(w)
        print(syllables[:200])
        unique_syllables = set(syllables)
        syllable_dict = {}
        i = 0
        for s in unique_syllables:
            syllable_dict[i] = s
            i += 1
        print(syllable_dict)
        return syllables, syllable_dict, len(syllable_dict)


if __name__ == "__main__":
    get_data("data/limericks.csv")
