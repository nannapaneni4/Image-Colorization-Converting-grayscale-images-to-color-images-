import torch
import os
from colorize_data import ColorizeData
from basic_model import Net, U2Net
from torch.utils.data import DataLoader
from torch.nn import MSELoss, HuberLoss, L1Loss
from tqdm import tqdm
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Trainer:
    def __init__(self,bs,lr,epochs,loss_fn):
        self.bs = bs
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn
        pass
        # Define hparams here or load them from a config file
    def train(self,train_df,val_df):
        pass
        # dataloaders
        train_dataset = ColorizeData(train_df)
        train_dataloader = DataLoader(train_dataset, batch_size = self.bs)
        val_dataset = ColorizeData(val_df)
        val_dataloader = DataLoader(val_dataset, batch_size = self.bs)
        # Model
        model = Net()
        # Loss function to use 
        try:
            if self.loss_fn=='mse':
                criterion = MSELoss()
            elif self.loss_fn=='huber':
                criterion = HuberLoss()
            elif self.loss_fn=='mae':
                criterion = L1Loss()
        except:
            raise ValueError("Loss function {} is invalid".format(self.loss_fn))
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # train loop
        model = model.to(device)
        model.train()
        loss_arr=[]
        min_val_loss = float('inf')
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in tqdm(train_dataloader):
                input = batch[0].to(device)
                output = model(input)
                target = batch[1].to(device)
                loss = criterion(output, target)
                optimizer.zero_grad()
                epoch_loss+=loss.item()
                loss.backward()
                optimizer.step()
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print("Loss at epoch {} = {}".format(epoch, avg_epoch_loss))
            loss_arr.append(avg_epoch_loss)
            if epoch %5 == 4:
                val_loss = self.validate(model, criterion, val_dataloader)
                print("Validation error at epoch {} is {}".format(epoch, val_loss))
                if val_loss < min_val_loss:
                    # checkpointing only when the val loss is decreasing
                    print("Val loss less at epoch {}. Saving model".format(epoch))
                    torch.save(model, 'basic_coloriser.model')
                    min_val_loss = val_loss
                model.train()
        print('End of Training')


    def validate(self, model, criterion, val_dataloader):
        pass
        # Validation loop begin
        model.eval()
        error = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                input = batch[0].to(device)
                output = model(input)
                target = batch[1].to(device)
                error += criterion(output, target)
        return error/len(val_dataloader)
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.

class U2NetTrainer:
    def __init__(self,bs,lr,epochs,loss_fn):
        self.bs = bs
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn
        pass
        # Define hparams here or load them from a config file
    def train(self,train_df,val_df):
        pass
        # dataloaders
        train_dataset = ColorizeData(train_df)
        train_dataloader = DataLoader(train_dataset, batch_size = self.bs)
        val_dataset = ColorizeData(val_df)
        val_dataloader = DataLoader(val_dataset, batch_size = self.bs)
        # Model
        model = U2Net()
        # Loss function to use 
        try:
            if self.loss_fn=='mse':
                criterion = MSELoss()
            elif self.loss_fn=='huber':
                criterion = HuberLoss()
            elif self.loss_fn=='mae':
                criterion = L1Loss()
        except:
            raise ValueError("Loss function {} is invalid".format(self.loss_fn))
        # You may also use a combination of more than one loss function 
        # or create your own.
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # train loop
        model = model.to(device)
        model.train()
        loss_arr=[]
        min_val_loss = float('inf')
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in tqdm(train_dataloader):
                input = batch[0].to(device)
                output = model(input)
                target = batch[1].to(device)
                loss = criterion(output, target)
                optimizer.zero_grad()
                epoch_loss+=loss.item()
                loss.backward()
                optimizer.step()
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print("Loss at epoch {} = {}".format(epoch, avg_epoch_loss))
            loss_arr.append(avg_epoch_loss)
            if epoch %5 == 4:
                val_loss = self.validate(model, criterion, val_dataloader)
                print("Validation error at epoch {} is {}".format(epoch, val_loss))
                if val_loss < min_val_loss:
                    # checkpointing only when the val loss is decreasing
                    print("Val loss less at epoch {}. Saving model".format(epoch))
                    torch.save(model, 'U2Net_coloriser.model')
                    min_val_loss = val_loss
                model.train()
        print('End of Training')

    def validate(self, model, criterion, val_dataloader):
        pass
        # Validation loop begin
        model.eval()
        loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                input = batch[0].to(device)
                output = model(input)
                target = batch[1].to(device)
                loss += criterion(output, target)
        return loss / len(val_dataloader)
        # Validation loop end