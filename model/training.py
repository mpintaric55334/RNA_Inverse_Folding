from model.model import RNAModel


def training_loop(model, trainloader, loss_class, optimizer, epoch_num=10):

    #  note to self: add gpu
    model.train()

    print(len(trainloader))
    for epoch in range(epoch_num):

        epoch_loss = 0
        n = 0
        for matrices, sequences, masks in trainloader:

            optimizer.zero_grad()

            outputs = model(matrices, sequences)
            loss = loss_class.compute_loss(outputs, sequences, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1
            print("ELoss is:", loss.item())

        print("Epoch", epoch, "average loss is:", epoch_loss/n)
