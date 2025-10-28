

import time
import os
import matplotlib.pyplot as plt                     
import numpy as np
import torch
torch.cuda.is_available()
   
import torch.nn as nn       
import torch.nn.functional as F
from monai import transforms
from torch.amp import autocast
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader 
from torch.nn import BCEWithLogitsLoss                    
from metrics.Dice import DiceCoefficient    
import sys   
from Dice import DiceCoefficient
from networks.nets.attention_3Dunet import UNet3D
from Experiments.data_reader import CTScanDataset, ToTensor

torch.multiprocessing.set_sharing_strategy("file_system")

# Define the combined BCE and Dice loss class
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.0
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return bce_loss + dice_loss

# Instantiate the combined loss function
combined_loss_function = BCEDiceLoss()

class Experiment_transformation():
    def __init__(self, args):
        for key, value in args.items():
            setattr(self, key, value)
            
        self.train_dir = args['train_dir']
        self.val_dir = args['val_dir']
        self.random_seed = args['random_seed']
        self.batch_size = args['batch_size']
        self.num_train_timesteps = args['num_train_timesteps']
        self.lr = args['lr']
        self.n_epochs = args['n_epochs']
        self.val_interval = args['val_interval']
        self.save_each_epoch = args['save_each_epoch']
        self.val_loader = None
        self.train_loader = None
        self.test_loader = None

        if args['run_mode'] == 'GPU':
            self.device = torch.device("cuda")
            print("Using GPU.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU.")

    def prepare(self):
        train_transformed_dataset = CTScanDataset(ct_dir='/content/drive/MyDrive/CTLung/images',
                                                  seg_dir='/content/drive/MyDrive/CTLung/masks',
                                                  transform=transforms.Compose([ToTensor()]))

        #val_transformed_dataset = CTScanDataset(ct_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/All Data/val',
                                               # seg_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/All Data/seg_val',
                                                #transform=transforms.Compose([ToTensor()]))

        #test_transformed_dataset = CTScanDataset(ct_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Test_data/CT',
                                                # seg_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Test_data/CT-Segmentation',
                                                # transform=transforms.Compose([ToTensor()]))

        self.train_loader = DataLoader(train_transformed_dataset, batch_size=1, shuffle=True, num_workers=4)
        #self.val_loader = DataLoader(val_transformed_dataset, batch_size=1, shuffle=True, num_workers=4)
        #self.test_loader = DataLoader(test_transformed_dataset, batch_size=1, shuffle=False, num_workers=4)

        print(f"Number of training samples: {len(train_transformed_dataset)}")
        #print(f"Number of validating samples: {len(val_transformed_dataset)}")
        #print(f"Number of test samples: {len(test_transformed_dataset)}")
        
    def plot_list(self, value_list, title, xlabel, ylabel, out_path):
        plt.figure(figsize=(8, 5))
        plt.title(title)
        plt.plot(value_list)
        
        plt.xticks(range(len(value_list)), [str(i) for i in range(len(value_list))], rotation=45)

        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.savefig(out_path, dpi=300)
        plt.close()

    def load_checkpoint(self, model, optimizer, checkpoint_path):
      if os.path.isfile(checkpoint_path):
          try:
              checkpoint = torch.load(checkpoint_path, map_location=self.device)
              model.load_state_dict(checkpoint['model_state_dict'])
              optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
              start_epoch = checkpoint['epoch']
              best_loss = checkpoint['best_loss']
              print(f"Checkpoint loaded successfully from {checkpoint_path}")
          except Exception as e:
              print(f"Error loading checkpoint: {e}")
              return model, optimizer, 0, float('inf')
      else:
          raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
      
      return model, optimizer, start_epoch, best_loss

    def save_checkpoint(self, model, optimizer, epoch, val_loss):
        checkpoint_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'model.pth'))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': val_loss
        }
        torch.save(checkpoint, checkpoint_model_path)

    def train(self):
            model = UNet3D        
            model.to(self.device)

            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
            epoch_loss_list = []
            val_epoch_loss_list = []
            val_dice_score_list = []
            test_epoch_loss_list = []
            test_dice_score_list = []
            
            scaler = GradScaler()
            total_start = time.time()

            checkpoint_files = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'model.pth'))
            start_epoch = 1

            if os.path.isfile(checkpoint_files):
              try:
                print(f'loading model from {checkpoint_files}')
                model, optimizer, start_epoch, best_loss = self.load_checkpoint(model, optimizer, checkpoint_files)
                start_epoch +=1
              except Exception as e:
                print(f"Error loading checkpoint: {e}")
            else:
              print(f"Checkpoint file not found at: {checkpoint_files}")

            for epoch in range(start_epoch, self.n_epochs):
                model.train()
                epoch_loss = 0
                dice_scores = []
                best_val_dice = 0.0
                best_epoch = 0
                for step, data in enumerate(self.train_loader):         
                    images = data["ct"].to(self.device)
                    seg = data["seg"].to(self.device)
            
                    batch_size = images.size(0)
                    optimizer.zero_grad(set_to_none=True)
                    
                    with autocast(device_type='cuda', enabled=True):
                        prediction = model(x=images)

                        if prediction.shape[-2:] != seg.shape[-2:]:
                            prediction = F.interpolate(prediction, size=seg.shape[-2:], mode='nearest')
                        
                        prediction_dice = torch.sigmoid(prediction)
                        prediction_dice = (prediction_dice > 0.5).float()
                        
                        dice_score = DiceCoefficient(prediction_dice, seg)
                        dice_scores.append(dice_score.cpu())
                        
                        loss = combined_loss_function(prediction, seg)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item() * batch_size

                avg_train_loss = epoch_loss / len(self.train_loader.dataset)
                avg_train_dice = np.mean(dice_scores)
            
                print(f'Epoch {epoch} Train loss {avg_train_loss:.4f} Train Dice {avg_train_dice:.4f}')

                epoch_folder = os.path.join('output', f'epoch_{epoch}')
                os.makedirs(epoch_folder, exist_ok=True)

                model_weights_path = os.path.join(epoch_folder, 'model.pth')
                torch.save(model.state_dict(), model_weights_path)

                train_losses_path = os.path.join(epoch_folder, 'train_losses.png')
                self.plot_list(epoch_loss_list, title='Training Loss in each epoch',
                              xlabel='Epoch', ylabel='Loss Value',
                              out_path=train_losses_path)

                if epoch % self.val_interval == 0:
                    model.eval()
                    val_epoch_loss = 0
                    best_val_dice = 0.0
                    best_epoch = 0
                    val_dice_scores = []
                    for step, data_val in enumerate(self.val_loader):
                        images = data_val["ct"].to(self.device)
                        seg = data_val["seg"].to(self.device)
                        batch_size = images.size(0)
                        with torch.no_grad():
                            with autocast(device_type='cuda', enabled=True):     
                                prediction = model(x=images)
                
                                if prediction.shape[-2:] != seg.shape[-2:]:
                                    prediction = F.interpolate(prediction, size=seg.shape[-2:], mode='nearest')
                                
                                prediction_dice = torch.sigmoid(prediction)
                                prediction_dice = (prediction_dice > 0.5).float()
                                
                                dice = DiceCoefficient(prediction_dice, seg)
                                val_dice_scores.append(dice.cpu())
                                
                                loss = combined_loss_function(prediction, seg)
                                
                            val_epoch_loss += loss.item() * batch_size
                        
                    avg_val_loss = val_epoch_loss / len(self.val_loader)
                    avg_val_dice = np.mean(val_dice_scores)
                
                    print("Epoch", epoch, "Validation loss", avg_val_loss, "Val dice", avg_val_dice.item())
                    val_epoch_loss_list.append(avg_val_loss)
                    val_dice_score_list.append(avg_val_dice)

                    if avg_val_dice > best_val_dice:
                        best_val_dice = avg_val_dice
                        best_epoch = epoch
                        self.save_checkpoint(model, optimizer, epoch, avg_val_loss)
                    print(f"Best Epoch: {best_epoch} with Validation Dice Score: {best_val_dice:.4f}")    

                    val_losses_path = os.path.join(epoch_folder, 'val_losses.png')
                    self.plot_list(
                                  value_list=val_epoch_loss_list,
                                  title='Validation loss in each epoch',
                                  xlabel='Epoch',
                                  ylabel='Loss Value',
                                  out_path=val_losses_path
                                  )

                    dice_scores_path = os.path.join(epoch_folder, 'dice_scores.png')
                    self.plot_list(
                                  value_list=val_dice_score_list,
                                  title='Dice score in each epoch',
                                  xlabel='Epoch',
                                  ylabel='Dice Score',
                                  out_path=dice_scores_path
                                  )   

                    out_path = os.path.join('output', f'epoch_{epoch}')
                    os.makedirs(out_path, exist_ok=True)

                    self.save_checkpoint(model, optimizer, epoch, avg_val_loss)

                self.plot_list(val_epoch_loss_list, title='Validation loss in each epoch',
                                  xlabel='Epoch', ylabel='Loss Value',
                                  out_path=os.path.join(out_path, 'val_losses.png'))

                self.plot_list(
                        value_list=val_dice_score_list,
                        title='Dice score at the end of epoch',
                        xlabel='Epoch',
                        ylabel='Dice Score',
                        out_path=dice_scores_path
                    )

                # Test
                model.eval()
                test_epoch_loss = 0
                test_dice_scores = []
                for step, data_test in enumerate(self.test_loader):
                    images = data_test["ct"].to(self.device)
                    seg = data_test["seg"].to(self.device)
                    batch_size = images.size(0)

                    with torch.no_grad():
                        with autocast(enabled=True):
                            prediction = model(x=images)

                            if prediction.shape[-2:] != seg.shape[-2:]:
                                prediction = F.interpolate(prediction, size=seg.shape[-2:], mode='nearest')

                            prediction_dice = torch.sigmoid(prediction)
                            prediction_dice = (prediction_dice > 0.5).float()

                            dice = DiceCoefficient(prediction_dice, seg)
                            test_dice_scores.append(dice.cpu())

                            loss = combined_loss_function(prediction, seg)

                    test_epoch_loss += loss.item() * batch_size

                avg_test_loss = test_epoch_loss / len(self.test_loader)
                avg_test_dice = np.mean(test_dice_scores)

                print(f'Test loss {avg_test_loss:.4f} Test Dice {avg_test_dice:.4f}')
                test_epoch_loss_list.append(avg_test_loss)
                test_dice_score_list.append(avg_test_dice)

                test_losses_path = os.path.join('output', 'test_losses.png')
                self.plot_list(
                    value_list=test_epoch_loss_list,
                    title='Test loss in each epoch',
                    xlabel='Epoch',
                    ylabel='Loss Value',
                    out_path=test_losses_path
                )

                test_dice_scores_path = os.path.join('output', 'test_dice_scores.png')
                self.plot_list(
                    value_list=test_dice_score_list,
                    title='Test Dice score in each epoch',
                    xlabel='Epoch',
                    ylabel='Dice Score',
                    out_path=test_dice_scores_path
                )

            checkpoint_last = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'last.pth'))
            torch.save(model.state_dict(), checkpoint_last)

            total_time = time.time() - total_start
            print(f"Training and testing model completed, total time: {total_time}.")


