import torch
from torch.utils.data import Dataset
import numpy as np
import os
from data.bpseq_loading.tokenizer import Tokenizer


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

            if len(stripped_line) == 3:  # Position, nucleotide,paired position
                pos, nucleotide, paired_pos = stripped_line
                pos, paired_pos = int(pos)-1, int(paired_pos) - 1  # -1 is because bpseq start index counting from 1

                nucleotide = nucleotide.upper()              
                if nucleotide not in "ACGU":
                    if nucleotide == "T":
                        nucleotide = "U"
                    else:
                        nucleotide = "N"
                sequence += nucleotide

                if (paired_pos != -1):
                    pairs.append([pos, paired_pos])

    length = len(sequence)
    matrix = torch.zeros((length, length), dtype=torch.float32)
    for pair_index1, pair_index2 in pairs:
        matrix[pair_index1][pair_index2] = 1
    return matrix, sequence


class BPSeqDataset(Dataset):

    """
    Custom torch dataset containing position matrices and appropriate sequences

    """

    def __init__(self, directory_path: str, cuttof_size: int = 256):
        self.directory_path = directory_path
        self.matrices = []
        self.sequences = []
        self.masks = []

        LONGEST_SEQUENCE = cuttof_size  # decided for computation purposes
        
        # loading initial matrices
        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            if os.path.isfile(file_path):

                matrix, sequence = parse_bpseq_file(file_path)
                
                if len(sequence) > LONGEST_SEQUENCE:
                    continue

                # remove all sequences with N > 30% of sequence
                N_count = 0

                for nucleotide in sequence:
                    if nucleotide == "N":
                        N_count += 1
                if N_count > len(sequence) * 0.3:
                    continue
                
                # padding
                padded_matrix = torch.zeros((LONGEST_SEQUENCE,
                                             LONGEST_SEQUENCE),
                                            dtype=torch.float32)
                matrix_size = matrix.shape[0]
                padded_matrix[:matrix_size, :matrix_size] = matrix

                padded_matrix = padded_matrix.reshape(1, LONGEST_SEQUENCE,
                                                      LONGEST_SEQUENCE)  # add channel for convolution
                self.matrices.append(padded_matrix)

                padded_sequence = sequence

                for _ in range(len(sequence), LONGEST_SEQUENCE):
                    padded_sequence += "P"  # P is used as padding

                tokenizer = Tokenizer()
                embedding_array = tokenizer.embedd(padded_sequence)

                self.sequences.append(embedding_array)

                mask = []
                for element in embedding_array:
                    if element != 5 and element != 6:  # token for N
                        mask.append(1)
                    else:
                        mask.append(0)

                mask = mask[1:]  # first element is unimportant, since its start token
                mask = np.array(mask)

                self.masks.append(mask)

    def __len__(self):

        return len(self.matrices)

    def __getitem__(self, idx):

        return self.matrices[idx], self.sequences[idx], self.masks[idx]
