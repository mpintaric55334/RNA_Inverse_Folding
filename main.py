from model.model import RNAModel
from model.train_and_infer import training_loop, evaluate, dataset_split
from utils.loss import Loss
from data.bpseq_loading.dataset import BPSeqDataset
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import wandb
from pathlib import Path


dataset = BPSeqDataset("/home/mpintaric/RNA_FOLDING/data/bpRNA_1m_90_BPSEQLFILES", 80)
print("Loaded data, starting training")
print(len(dataset))
train, val, test = dataset_split(dataset, 6000, 500, 2517)

train_loader = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=1, shuffle=False)
test_loader = DataLoader(test, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNAModel(batch_size=32, matrix_shape=80, hidden_size=512,
                 encoder_channels=256).to(device)

loss_class = Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


model = training_loop(model, train_loader, val_loader, loss_class, optimizer,
                      lr=1e-3, epoch_num=100, device=device)
test_acc_non_pad = evaluate(model, test_loader, device, save=True)
print("Test accuracy not padded:", test_acc_non_pad)
wandb.log({"test_acc_non_pad": test_acc_non_pad})

model_path = Path("models/model2")

torch.save(model.state_dict(), model_path)
