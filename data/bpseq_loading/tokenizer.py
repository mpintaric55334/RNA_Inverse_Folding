import numpy as np

class Tokenizer:

    def __init__(self):
        self.lookup = {"A":0,"C":1,"G":2,"U":3,"N":4,"P":5}

    def embedd(self,sequence):
        embedding_array = np.array([],dtype=int)

        for nucleotide in sequence:
            embedding_array= np.append(embedding_array,self.lookup[nucleotide])


        return embedding_array