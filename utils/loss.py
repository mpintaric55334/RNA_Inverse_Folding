import torch.nn as nn
import torch


class Loss:

    def __init__(self):

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, predictions, true_sequence, masks):

        # reshaping so loss is easily calculable
        masks = masks.reshape(-1)

        predictions = predictions[:, :-1, :]  # remove last element because its EOS
        predictions = predictions.reshape(-1, predictions.shape[2])

        true_sequence = true_sequence[:, 1:]  # remove first element <s>
        true_sequence = true_sequence - 1  # have to remove one, because token 1 corresponds to index 0 in probability
        true_sequence = true_sequence.reshape(-1)

        loss = self.criterion(predictions, true_sequence)

        loss = loss * masks  # element wise for masking N

        return loss.sum() / masks.sum()  # divide by sum instead of mean because of 0
