from model.model import RNAModel
import wandb
import torch
from torch.utils.data import random_split
from utils.folding import save_position_matrix, save_sequence, save_sequence_true


def dataset_split(dataset, train_size, val_size, test_size):

    train_val, test = random_split(dataset, [train_size + val_size, test_size])
    train, val = random_split(train_val, [train_size, val_size])
    return train, val, test


def evaluate(model, testloader, device="cpu", save=False):

    model.eval()
    model.reset_batch_size(1)
    model.reset_states()

    matches_non_padded = 0
    total_non_padded = 0

    name = 0

    for matrices, sequences, masks, true_lengths in testloader:
        matrices = matrices.to(device)
        sequences = sequences.to(device)
        masks = masks.to(device)
        true_lengths = true_lengths.to(device)

        next_tokens = torch.zeros((sequences.shape[0], 1), device=device)
        predicted_sequences = torch.zeros((sequences.shape[0],
                                           sequences.shape[1]), device=device)
        for idx in range(1, true_lengths + 1):  # true lengths + 1 because we dont count start token
            outputs = model.infer(matrices, next_tokens)
            device = outputs.device
            next_tokens = outputs.argmax(dim=2) + 1   # start token masking
            next_tokens = next_tokens.to(device)
            predicted_sequences[:, idx] = next_tokens.squeeze(1)

        model.reset_states()
        predicted_sequences = predicted_sequences.long()

        predicted_sequences = predicted_sequences[:, :true_lengths + 1]
        sequences = sequences[:, :true_lengths + 1]
        matches_non_padded += (sequences[:, 1:] == predicted_sequences[:, 1:]).sum().item()
        total_non_padded += predicted_sequences.shape[1]

        if save:
            sequences_save = predicted_sequences.squeeze(0)[1:(true_lengths+1)]
            true_sequences_save = sequences.squeeze(0)[1:(true_lengths+1)]
            matrices = matrices.squeeze((0, 1))[:true_lengths, :true_lengths]

            save_sequence(sequences_save, str(name), "output_dir")
            save_sequence_true(true_sequences_save, str(name), "output_dir")
            matrices = matrices.squeeze((0, 1))[:true_lengths, :true_lengths]
            save_position_matrix(str(name), matrices, "output_dir")
            name += 1

    print(predicted_sequences[0])
    print(sequences[0])
    accuracy_non_padded = matches_non_padded/total_non_padded

    return accuracy_non_padded


def training_loop(model, trainloader, valloader, loss_class, optimizer,
                  lr=1e-4, epoch_num=10, device="cpu"):

    wandb.init(
        project="seminar_dipl",

        config={
         "learning_rate": lr,
         "epochs": epoch_num,
        }
    )

    print("Starting training on device", device)

    for epoch in range(epoch_num):

        model.train()
        epoch_loss = 0
        n = 0
        for matrices, sequences, masks, _ in trainloader:

            matrices = matrices.to(device)
            sequences = sequences.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(matrices, sequences)
            loss = loss_class.compute_loss(outputs, sequences, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1
            if n % 20 == 0:
                print("Batch", n, "/", len(trainloader), " loss of epoch",
                      epoch, " is", loss.item())

        print("Epoch", epoch, "average loss is:", epoch_loss/n)
        old_batch_size = model.batch_size
        val_accuracy_non_padded = evaluate(model, valloader, device)
        model.reset_batch_size(old_batch_size)
        model.reset_states()
        print("Epoch", epoch, " non padded validation accuracy is:",
              val_accuracy_non_padded)
        wandb.log({"loss": epoch_loss/n, "val_accuracy_non_padded":
                   val_accuracy_non_padded})

    return model
