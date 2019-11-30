import numpy as np
import csv
import string
import pronouncing


def get_data(file_path):
    # add any characters to this set if you don't want it in the final data
    exclude = set(['!', '.', ',', '?'])
    def flatten(l): return [item for sublist in l for item in sublist]
    with open(file_path) as f:
        csv_reader = csv.reader(f)
        data = []
        for row in csv_reader:
            # get rid of special chacacters and punctuation
            row[0] = "".join(c for c in row[0] if c not in exclude)
            # split by line
            poem = (row[0].split("\n"))[:-1]
            # double check the poem has 5 lines
            if len(poem) == 5:
                # each line is now split into words
                for i in range(5):
                    poem[i] = poem[i].split(" ")
                    # for each word:
                    for w in range(len(poem[i])):
                        # search word in cmudict
                        p_list = pronouncing.phones_for_word(poem[i][w])
                        # if syllables exist in dict, get the first one
                        if(len(p_list) > 0):
                            poem[i][w] = p_list[0]
                        # else, make an UNK syllable list of length
                        # where length is word length divided by 3
                        # [:-1] removes last space in UNK list
                        else:
                            poem[i][w] = (
                                "UNK " * int(len(poem[i][w]) / 3))[:-1]

                        # split the word by syllables
                        if w == (len(poem[i]) - 1):
                            # if last line, and last word, add END
                            if(i == 4):
                                poem[i][w] = (poem[i][w] + " END").split(" ")
                            # if not last line, but last word, add NEWLINE
                            else:
                                poem[i][w] = (
                                    poem[i][w] + " NEWLINE").split(" ")
                        # if not last line or last word, add SPACE
                        else:
                            poem[i][w] = (poem[i][w] + " SPACE").split(" ")
                    # make syllable word list into just a syllable list
                    # essentially, flatten the list
                    poem[i] = [item for sublist in poem[i]
                               for item in sublist]
                # flatten the list of newlines
                poem = [item for sublist in poem
                        for item in sublist]
                data.append(poem)
        print(data[0])
        print(len(data))
        return data


if __name__ == "__main__":
    get_data("data/limericks.csv")
