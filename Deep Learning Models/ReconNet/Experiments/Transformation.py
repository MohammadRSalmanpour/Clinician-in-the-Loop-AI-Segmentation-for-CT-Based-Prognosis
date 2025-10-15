# -*- coding: utf-8 -*-

# https://github.com/BITEWKRER/Lung-Nodule-Segmentation-Framework.git

import time
import os

import monai
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, list_data_collate
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from glob import glob
from torch.utils.data import DataLoader

import torch.nn.functional as F
# dice and loss
from torch.nn import BCEWithLogitsLoss
from metrics.Dice import DiceCoefficient
# import model
from networks.nets.ReconNet import ReconNet
from Experiments.data_reader_new import CTScanDataset, ToTensor
torch.cuda.empty_cache()
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.multiprocessing.set_sharing_strategy("file_system")

##########################################################New loss 
import torch
import torch.nn as nn

class CombinedDiceBCELoss(nn.Module):
    """
    Combines Dice Loss and Binary Cross-Entropy (BCE) Loss for 3D segmentation tasks.
    """
    def __init__(self, weight_dice=0.5, weight_bce=0.5, smooth=1.0, pos_weight=None):
        """
        Args:
            weight_dice (float): Weight for the Dice Loss component.
            weight_bce (float): Weight for the BCE Loss component.
            smooth (float): Smoothing factor for Dice Loss.
            pos_weight (torch.Tensor or None): Positive class weight for BCE Loss.
        """
        super(CombinedDiceBCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def dice_loss(self, prediction, target):
        """
        Computes Dice Loss using the dice_coef_3d function.
        Args:
            prediction (torch.Tensor): Predicted tensor of shape [batch, channel, depth, height, width].
            target (torch.Tensor): Ground truth tensor of the same shape as prediction.
        Returns:
            torch.Tensor: Dice Loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        prediction_prob = torch.sigmoid(prediction)
        dice_score = DiceCoefficient(target, prediction_prob, smooth=self.smooth)
        return 1 - dice_score  # Dice Loss = 1 - Dice Coefficient

    def forward(self, prediction, target):
        """
        Calculates the combined loss.
        Args:
            prediction (torch.Tensor): Predicted logits of shape [batch, channel, depth, height, width].
            target (torch.Tensor): Ground truth tensor of the same shape as prediction.
        Returns:
            torch.Tensor: Combined loss value.
        """
        # Calculate BCE Loss
        bce_loss = self.bce_loss(prediction, target)
        # Calculate Dice Loss
        dice_loss = self.dice_loss(prediction, target)
        # Combine the losses
        combined_loss = self.weight_dice * dice_loss + self.weight_bce * bce_loss
        return combined_loss



combined_loss_function = CombinedDiceBCELoss(weight_dice=0.5, weight_bce=0.5, smooth=1.0, pos_weight=torch.tensor(2.0))


########################################################################


class Experiment_transformation():
    def __init__(self, args):
        # self.prepare()

        for key, value in args.items():
            setattr(self, key, value)

        # data directory
        self.train_dir = args['train_dir']
        self.val_dir = args['val_dir']

        self.random_seed = args['random_seed']

        # data loader parameters
        self.batch_size = args['batch_size']

        # model hyperparameters
        self.num_train_timesteps = args['num_train_timesteps']
        self.lr = args['lr']
        self.n_epochs = args['n_epochs']
        self.val_interval = args['val_interval']

        # run mode
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

        # self.dice_metric = DiceCoefficient(reduction='mean')

    def prepare(self):

        # x, y = 256, 256

        train_transformed_dataset = CTScanDataset(ct_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_train/ROI_cropped/Cropped_CT',
                                                  seg_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_train/ROI_cropped/Cropped_seg',
                                                  transform=transforms.Compose([ToTensor()]))

        val_transformed_dataset = CTScanDataset(ct_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_VAl/ROI_cropped/Cropped_CT',
                                                seg_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_VAl/ROI_cropped/Cropped_seg',
                                                transform=transforms.Compose([ToTensor()]))

        test_transformed_dataset = CTScanDataset(ct_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_test/ROI_cropped/Cropped_CT',
                                                 seg_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_test/ROI_cropped/Cropped_seg',
                                                 transform=transforms.Compose([ToTensor()]))


        self.train_loader = DataLoader(train_transformed_dataset, batch_size=1, shuffle=True, num_workers=2,pin_memory=True)
        self.val_loader = DataLoader(val_transformed_dataset, batch_size=1, shuffle=True, num_workers=2,pin_memory=True)
        self.test_loader = DataLoader(test_transformed_dataset, batch_size=1, shuffle=False, num_workers=2,pin_memory=True)

        print(f"Number of training samples: {len(train_transformed_dataset)}")
        print(f"Number of validating samples: {len(val_transformed_dataset)}")
        print(f"Number of test samples: {len(test_transformed_dataset)}")

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
        if os.path.isfile(checkpoint_path):  # Check if the checkpoint file exists
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)  # Load checkpoint to the correct device
                model.load_state_dict(checkpoint['model_state_dict'])
                if optimizer:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Added this line
                start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                print(f"Checkpoint loaded successfully from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return model, optimizer, 0, float('inf')  # Return default values
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        return model, optimizer, start_epoch, best_loss

    def save_checkpoint(self, model, optimizer, epoch, val_loss):
        checkpoint_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output5e-5', 'model.pth'))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': val_loss
        }
        torch.save(checkpoint, checkpoint_model_path)

    def train(self):
        # Define Model with its HyperParameters
        model = ReconNet(32, 1)
        model.to(self.device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
        epoch_loss_list = []
        val_epoch_loss_list = []
        val_dice_score_list = []
        test_epoch_loss_list = []  # Added for test loss
        test_dice_score_list = []  # Added for test dice score

        # Start training
        scaler = GradScaler()
        total_start = time.time()

        checkpoint_files = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output5e-5', 'model.pth'))
        start_epoch = 1

        # Check if there are any checkpoint files
        if os.path.isfile(checkpoint_files):
            try:
                print(f'loading model from {checkpoint_files}')
                # Load the latest checkpoint
                model, optimizer, start_epoch, best_loss = self.load_checkpoint(model, optimizer, checkpoint_files)
                # Start from the next epoch
                start_epoch += 1
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        else:
            print(f"Checkpoint file not found at: {checkpoint_files}")

        # train
        for epoch in range(start_epoch, self.n_epochs):
            model.train()
            epoch_loss = 0
            dice_scores = []
            best_val_dice = 0.0
            best_epoch = 0

            for step, data in enumerate(self.train_loader):
                images = data["ct"].to(self.device)
                seg = data["seg"].to(self.device)  # this is the ground truth segmentation
                seg = (seg > 0.0).float()

                batch_size = images.size(0)
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    prediction = model(x=images)  # Change input channel to 1

                    # Resize prediction if necessary
                    if prediction.shape[-2:] != seg.shape[-2:]:
                        prediction = F.interpolate(prediction, size=seg.shape[-2:], mode='nearest')

                    prediction_dice = torch.sigmoid(prediction)
                    prediction_dice = (prediction_dice > 0.5).float()

                    dice_score = DiceCoefficient(prediction_dice, seg)
                    dice_scores.append(dice_score.cpu())

                    # Calculate the combined BCE and Dice loss
                    loss = combined_loss_function(prediction, seg)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item() * batch_size

            # Calculate average loss and Dice score for the epoch
            avg_train_loss = epoch_loss / len(self.train_loader.dataset)
            avg_train_dice = torch.mean(torch.stack(dice_scores)).item()#####@##################

            # Logging the results
            print(f'Epoch {epoch} Train loss {avg_train_loss:.4f} Train Dice {avg_train_dice:.4f}')

            # Save validation loss plot and dice score plot for the current epoch
            epoch_folder = os.path.join('output5e-5', f'epoch_{epoch}')
            os.makedirs(epoch_folder, exist_ok=True)

            # Save model weights for the current epoch
            model_weights_path = os.path.join(epoch_folder, 'model.pth')
            torch.save(model.state_dict(), model_weights_path)

            # Save training loss plot for the current epoch
            train_losses_path = os.path.join(epoch_folder, 'train_losses.png')
            self.plot_list(epoch_loss_list, title='Training Loss in each epoch',
                           xlabel='Epoch', ylabel='Loss Value',
                           out_path=train_losses_path)

            # Validation
            if epoch % self.val_interval == 0:
                model.eval()
                val_epoch_loss = 0
                best_val_dice = 0.0
                best_epoch = 0
                val_dice_scores = []

                for step, data_val in enumerate(self.val_loader):
                    images = data_val["ct"].to(self.device)
                    seg = data_val["seg"].to(self.device)  # this is the ground truth segmentation
                    seg = (seg > 0.0).float()
                    batch_size = images.size(0)
                    with torch.no_grad():
                        with autocast(enabled=True):
                            prediction = model(x=images)

                            # Resize prediction if necessary
                            if prediction.shape[-2:] != seg.shape[-2:]:
                                prediction = F.interpolate(prediction, size=seg.shape[-2:], mode='nearest')

                            prediction_dice = torch.sigmoid(prediction)
                            prediction_dice = (prediction_dice > 0.5).float()

                            dice = DiceCoefficient(prediction_dice, seg)
                            val_dice_scores.append(dice.cpu())

                            # Calculate the combined BCE and Dice loss
                            loss = combined_loss_function(prediction, seg)

                    val_epoch_loss += loss.item() * batch_size

                avg_val_loss = val_epoch_loss / len(self.val_loader)
                avg_val_dice = torch.mean(torch.stack(val_dice_scores)).item()

                print("Epoch", epoch, "Validation loss", avg_val_loss, "Val dice", avg_val_dice)
                val_epoch_loss_list.append(avg_val_loss)
                val_dice_score_list.append(avg_val_dice)

                if avg_val_dice > best_val_dice:
                    best_val_dice = avg_val_dice
                    best_epoch = epoch

                    # Save the model as the best model
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

                out_path = os.path.join('output5e-5', f'epoch_{epoch}')
                os.makedirs(out_path, exist_ok=True)

                # save model weights
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

            # Testing
            model.eval()
            test_epoch_loss = 0
            test_dice_scores = []

            for step, data_test in enumerate(self.test_loader):
                images = data_test["ct"].to(self.device)
                seg = data_test["seg"].to(self.device)  # Ground truth segmentation
                seg = (seg > 0.0).float()
                batch_size = images.size(0)

                with torch.no_grad():
                    with autocast(enabled=True):
                        prediction = model(x=images)

                        # Resize prediction if necessary
                        if prediction.shape[-2:] != seg.shape[-2:]:
                            prediction = F.interpolate(prediction, size=seg.shape[-2:], mode='nearest')

                        prediction_dice = torch.sigmoid(prediction)
                        prediction_dice = (prediction_dice > 0.5).float()

                        dice = DiceCoefficient(prediction_dice, seg)
                        test_dice_scores.append(dice.cpu())

                        # Calculate the combined BCE and Dice loss
                        test_loss = combined_loss_function(prediction, seg)

                test_epoch_loss += test_loss.item() * batch_size

            avg_test_loss = test_epoch_loss / len(self.test_loader)
            avg_test_dice = torch.mean(torch.stack(test_dice_scores)).item()

            print("Epoch", epoch, "Test loss", avg_test_loss, "Test dice", avg_test_dice)
            test_epoch_loss_list.append(avg_test_loss)
            test_dice_score_list.append(avg_test_dice)

            test_losses_path = os.path.join(epoch_folder, 'test_losses.png')
            self.plot_list(
                value_list=test_epoch_loss_list,
                title='Test loss in each epoch',
                xlabel='Epoch',
                ylabel='Loss Value',
                out_path=test_losses_path
            )

            test_dice_scores_path = os.path.join(epoch_folder, 'test_dice_scores.png')
            self.plot_list(
                value_list=test_dice_score_list,
                title='Test Dice score in each epoch',
                xlabel='Epoch',
                ylabel='Dice Score',
                out_path=test_dice_scores_path
            )

        checkpoint_last = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output5e-5', 'last.pth'))
        torch.save(model.state_dict(), checkpoint_last)

        total_time = time.time() - total_start
        print(f"Training and testing model completed, total time: {total_time}.")
