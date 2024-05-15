from model.model import RNAModel
from model.train_and_infer import training_loop, evaluate, dataset_split
from utils.loss import Loss
from data.bpseq_loading.dataset import BPSeqDataset
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import wandb


dataset = BPSeqDataset("/home/mpintaric/RNA_FOLDING/data/bpRNA_1m_90_BPSEQLFILES", 128)
print("Loaded data, starting training")

train, val, test = dataset_split(dataset, 15000, 569, 3000)

train_loader = DataLoader(train, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=16, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=16, shuffle=False, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNAModel(16, 128).to(device)

loss_class = Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


model = training_loop(model, train_loader, val_loader, loss_class, optimizer,
                      lr=1e-3, epoch_num=100, device=device)

test_acc, test_acc_non_pad = evaluate(model, test_loader, device)
print("Test accuracy:", test_acc)
print("Test accuracy not padded:", test_acc_non_pad)
wandb.log({"test_acc": test_acc, "test_acc_non_pad": test_acc_non_pad})
