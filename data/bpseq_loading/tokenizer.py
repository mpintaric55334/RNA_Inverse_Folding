import numpy as np


class Tokenizer:

    def __init__(self):
        # start_token = 0
        self.lookup = {"A": 1, "C": 2, "G": 3, "U": 4, "N": 5, "P": 6}

    def embedd(self, sequence):
        embedding_array = np.array([0], dtype=int)  # include start token

        for nucleotide in sequence:
            embedding_array = np.append(embedding_array,
                                        self.lookup[nucleotide])

        return embedding_array
