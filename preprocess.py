import numpy as np
import csv
import string

def get_data(file_path):
    # add any characters to this set if you don't want it in the final data
    exclude = set(['!', '.', ',', '?'])
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
                data.append(poem)
        print(data[0])
        print(len(data))
        return data

if __name__ == "__main__":
	get_data("data/limericks.csv")