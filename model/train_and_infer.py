from model.model import RNAModel
import wandb
import torch
from torch.utils.data import random_split


def dataset_split(dataset, train_size, val_size, test_size):

    train_val, test = random_split(dataset, [train_size + val_size, test_size])
    train, val = random_split(train_val, [train_size, val_size])
    return train, val, test


def evaluate(model, testloader, device="cpu"):

    model.eval()
    matches = 0
    total_nucleotides = 0
    for matrices, sequences, masks in testloader:

        matrices = matrices.to(device)
        sequences = sequences.to(device)
        masks = masks.to(device)
        next_tokens = torch.zeros((sequences.shape[0], 1)).to(device)
        predicted_sequences = torch.zeros((sequences.shape[0], 1)).to(device)

        for _ in range(1, sequences.shape[1]):
            outputs = model(matrices, next_tokens)
            next_tokens = outputs.argmax(dim=2) + 1   # start token masking
            next_tokens = next_tokens.to(device)
            predicted_sequences = torch.cat((predicted_sequences, next_tokens),
                                            dim=1)
            predicted_sequences = predicted_sequences.to(device)

        predicted_sequences = predicted_sequences.long()
        matches += (sequences == predicted_sequences).sum().item()
        total_nucleotides += sequences.shape[0] * sequences.shape[1]
    accuracy = matches/total_nucleotides
 
    return accuracy


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
        for matrices, sequences, masks in trainloader:

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
        val_accuracy = evaluate(model, valloader, device)
        print("Epoch", epoch, " validation accuracy is:", val_accuracy)
        wandb.log({"loss": epoch_loss/n, "val_accuracy": val_accuracy})

    return model
