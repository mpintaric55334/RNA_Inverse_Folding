import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tokenizer import Tokenizer

def parse_bpseq_file(filename: str):

    """
    Function to create position matrix and sequence from a bpseq file
    Arguments:
        filename: bpseq filename

    Returns:
        matrix: symmetrical matrix with pair indexes set to 1, others to 0
        sequence: string sequence
    """

    with open(filename, 'r') as file:

        sequence = ''
        pairs = []

        for line in file:
            if "#" in line: 
                continue
            stripped_line = line.strip().split()

            if len(stripped_line) == 3:  #Position, nucleotide, paired position
                pos, nucleotide, paired_pos = stripped_line
                pos,paired_pos = int(pos)-1, int(paired_pos)-1 #-1 is because bpseq start index counting from 1

                nucleotide = nucleotide.upper()
                
                if nucleotide not in "ACGU":
                    if nucleotide == "T":
                        nucleotide = "U"
                    else:
                        nucleotide = "N"
                
                sequence += nucleotide

                if(paired_pos != -1):
                    pairs.append([pos,paired_pos])

    length = len(sequence)
    matrix = np.zeros((length,length),dtype=int)
    for pair_index1, pair_index2 in pairs:
        matrix[pair_index1][pair_index2] = 1
    
    return matrix, sequence



class BPSeqDataset(Dataset):

    """
    Custom torch dataset containing position matrices and appropriate sequences

    """

    def __init__(self,directory_path: str):
        self.directory_path = directory_path
        self.matrices = []
        self.sequences = []

        LONGEST_SEQUENCE = 4381 #this we know from EDA
        
        #loading initial matrices
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            if os.path.isfile(file_path):

                matrix,sequence = parse_bpseq_file(file_path)

                #padding
                padded_matrix = np.zeros((LONGEST_SEQUENCE,LONGEST_SEQUENCE),dtype=int)
                matrix_size = matrix.shape[0]
                padded_matrix[:matrix_size,:matrix_size] = matrix

                self.matrices.append(padded_matrix)

                padded_sequence = sequence

                for i in range(len(sequence),LONGEST_SEQUENCE):
                    padded_sequence += "P" # P is used as padding

                tokenizer = Tokenizer()
                embedding_array = tokenizer.embedd(padded_sequence)

                self.sequences.append(embedding_array)


    def __len__(self):

        return len(self.matrices)

    def __getitem__(self,idx):

        return self.matrices[idx],self.sequences[idx]
