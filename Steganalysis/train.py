from torch import optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import BaselineModel, Baseline_enet,Model,initialize_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import StegDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import config
from config import binary_acc
import warnings
import sys

warnings.filterwarnings("ignore")


PATH = config.PATH
batch_size = int(sys.argv[-2])
lr = float(sys.argv[-1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


dataset = StegDataset(
    'csv_files/final_steganalysis_6000_samples.csv',
    norm=False,
    transform=config.transform
)

l = len(dataset)
tr_size = int(l*0.75)
val_size = (l-tr_size)


train_set, val_set = torch.utils.data.random_split(
    dataset, [tr_size, val_size])

train_loader = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0, drop_last=True)


val_loader = DataLoader(dataset=val_set,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=0)

epochs = config.NUM_EPOCHS
e_loss = []
e_val_loss = []
e_val_score = []
n_iter = 0
n_epochs_stop = config.EPOCH_THRES
epochs_no_improve = 0
early_stop = True
min_val_loss = np.Inf
max_val_score = 0
e = 0

model_ft = initialize_model('vgg',1,use_pretrained=True)
model = model_ft.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=config.WEIGHT_DECAY)

#print(model)

writer = SummaryWriter()
scheduler = StepLR(optimizer, step_size=config.STEP_SIZE,gamma=config.GAMMA, verbose=True)

criterion = nn.BCELoss()

for epoch in range(epochs):
    print('Epoch {}/{}, lr:{}'.format(epoch + 1,
          epochs, optimizer.param_groups[0]['lr']))
    print('-' * 10)

    running_loss = 0.0
    running_score = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        for (images, label) in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")

            images, label = images.to(device), label.to(device)

            output = model(images)
            label = label.unsqueeze(1).float()

            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())
            writer.add_scalar('Training Loss', loss, global_step=e)
        #scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    print(f"Train - Loss: {epoch_loss:.4f}")
    e_loss.append(epoch_loss)
    e += 1

    with torch.no_grad():
        val_loss = 0
        val_score = 0
        for batch_idx, (images_val, label_val) in enumerate(val_loader):
            images_val, label_val = images_val.to(device), label_val.to(device)

            output = model(images_val)
            
            loss = criterion(output, label)
            score = binary_acc(output, label_val)

            val_loss += loss.item()

            val_score += score.item()

        val_epoch_loss = val_loss / len(val_loader)
        print(f"Val Loss: {val_epoch_loss:.4f}")
        e_val_loss.append(val_epoch_loss)

        val_epoch_score = val_score/len(val_loader)
        print(f"Val Score: {val_epoch_score:.8f}")
        e_val_score.append(val_epoch_score)
        
        torch.save(model.state_dict(), PATH)
        print('Model Saved at:', str(PATH), "\n")
        """
        if val_epoch_loss < min_val_loss:
            print('val_loss<min_val_loss', min_val_loss)
            torch.save(model.state_dict(), PATH)
            print('Model Saved at:', str(PATH), '\n')
            epochs_no_improve = 0
            min_val_loss = val_epoch_loss
        else:
            epochs_no_improve += 1

        
        if val_epoch_score > max_val_score:
            print('val_score>max_val_score', max_val_score)
            torch.save(model.state_dict(), PATH)
            print('Model Saved at:', str(PATH),"\n")
            epochs_no_improve = 0
            max_val_score = val_epoch_score

        else:
            epochs_no_improve += 1
        
        n_iter += 1

        if epoch > 5 and epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            early_stop = True
            break
        else:
            continue
        """
plt.figure(figsize=(10, 8))
plt.plot(1+np.arange(e), e_loss, label='Training Loss')
plt.plot(1+np.arange(e), e_val_loss, label='Validation Loss')
plt.plot(1+np.arange(e), e_val_score,
         label='Validation Score(Accuracy)')
plt.legend()
plt.show()
