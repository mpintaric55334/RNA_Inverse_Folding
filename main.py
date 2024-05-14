from model.model import RNAModel
from model.train_and_infer import training_loop, evaluate, dataset_split
from utils.loss import Loss
from data.bpseq_loading.dataset import BPSeqDataset
import torch.optim as optim
import torch
from torch.utils.data import DataLoader


dataset = BPSeqDataset("/home/mpintaric/RNA_FOLDING/data/bpRNA_1m_90_BPSEQLFILES", 256)
print("Loaded data, starting training")

train, val, test = dataset_split(dataset, 21685, 100, 2900)

train_loader = DataLoader(train, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val, batch_size=64, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=64, shuffle=False, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNAModel(64, 256).to(device)

loss_class = Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


model = training_loop(model, train_loader, val_loader, loss_class, optimizer,
                      lr=1e-4, epoch_num=10, device=device)

test_acc = evaluate(model, test_loader, device)
print("Test accuracy:", test_acc)
