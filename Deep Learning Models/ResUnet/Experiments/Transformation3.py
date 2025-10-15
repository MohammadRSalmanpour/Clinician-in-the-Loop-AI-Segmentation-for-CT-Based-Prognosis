import time
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from metrics.Dice import DiceCoefficient
from networks.nets.ResUNet import ResUNet
from Experiments.data_reader_new import CTScanDataset, ToTensor, Resize

torch.multiprocessing.set_sharing_strategy("file_system")







###########################################################
# Added Metrics Functions
###########################################################
import torch
import numpy as np
from medpy.metric import hd

def hausdorff_distance(y_true, y_pred):
    """
    Compute Hausdorff Distance between binary segmentation masks using MedPy
    Args:
        y_true: Ground truth tensor of shape [batch, channel, depth, height, width]
        y_pred: Predicted tensor of same shape
    Returns:
        torch.Tensor: Hausdorff Distance
    """
    batch_hd = []
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    for i in range(y_true.shape[0]):
        true_mask = y_true[i, 0] > 0.5
        pred_mask = y_pred[i, 0] > 0.5
        
        if np.sum(true_mask) == 0 or np.sum(pred_mask) == 0:
            batch_hd.append(0.0)
            continue
        
        # Calculate Hausdorff Distance using MedPy
        hd_distance = hd(true_mask, pred_mask)
        batch_hd.append(hd_distance)
    
    return torch.tensor(np.mean(batch_hd))


def iou_score(y_true, y_pred, smooth=1.0):
    """
    Compute Intersection over Union (Jaccard Index)
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        smooth: Smoothing factor to avoid division by zero
    Returns:
        torch.Tensor: IoU score
    """
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def DiceCoefficient(target, prediction, smooth=1.0):
    """
    Compute Dice Coefficient
    Args:
        target: Ground truth tensor
        prediction: Predicted tensor
        smooth: Smoothing factor to avoid division by zero
    Returns:
        torch.Tensor: Dice score
    """
    intersection = torch.sum(target * prediction)
    union = torch.sum(target) + torch.sum(prediction)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

###########################################################


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
        for key, value in args.items():
            setattr(self, key, value)

        self.device = torch.device("cuda" if args['run_mode'] == 'GPU' else "cpu")
        print(f"Using {self.device}.")

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def prepare(self):
        target_size = (64,64,64)
    
        transform = transforms.Compose([Resize(target_size), ToTensor()])

        train_transformed_dataset = CTScanDataset(
            ct_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_train/ROI_cropped/Cropped_CT',
            seg_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_train/ROI_cropped/Cropped_seg',
            transform=transform
        )

        val_transformed_dataset = CTScanDataset(
            ct_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_VAl/ROI_cropped/Cropped_CT',
            seg_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_VAl/ROI_cropped/Cropped_seg',
            transform=transform
        )

        test_transformed_dataset = CTScanDataset(
            ct_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_test/ROI_cropped/Cropped_CT',
            seg_dir='/home/jhubadmin/.ssh/Projects/3D_Segmentation/Final_Data/step4_test/ROI_cropped/Cropped_seg',
            transform=transform
        )

        self.train_loader = DataLoader(train_transformed_dataset, batch_size=1, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_transformed_dataset, batch_size=1, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(test_transformed_dataset, batch_size=1, shuffle=False, num_workers=4)

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
        
        model = ResUNet()
        model.to(self.device)
        
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)#############################################
        epoch_loss_list = []
        val_epoch_loss_list = []
        val_dice_score_list = []
        test_dice_score_list = []
        dice_scores = []
        hd_scores = []
        iou_scores = []
        
        scaler = GradScaler()
        total_start = time.time()
        
        checkpoint_files = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'model.pth'))
        start_epoch = 1
        
        if os.path.isfile(checkpoint_files):
            try:
                print(f'loading model from {checkpoint_files}')
                model, optimizer, start_epoch, best_loss = self.load_checkpoint(model, optimizer, checkpoint_files)
                start_epoch += 1
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        else:
            print(f"Checkpoint file not found at: {checkpoint_files}")
        
        for epoch in range(start_epoch, self.n_epochs + 1):
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
                
                with autocast(enabled=True):
                    prediction = model(images)
                    
                    # Check the dimensions of the target segmentation mask
                    target_size = seg.shape[2:]
                    # Adjust interpolation to handle 3D tensors
                    prediction = F.interpolate(prediction, size=target_size, mode='trilinear', align_corners=False)
            
                    prediction_dice = torch.sigmoid(prediction)
                    prediction_dice = (prediction_dice > 0.5).float()
                    
                    dice_score = DiceCoefficient(prediction_dice, seg)
                    hd = hausdorff_distance(seg, prediction_dice)
                    iou = iou_score(seg, prediction_dice)
                    
                    dice_scores.append(dice_score.cpu())
                    hd_scores.append(hd)
                    iou_scores.append(iou)
                    
                    loss = combined_loss_function(prediction, seg)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item() * batch_size
            
            # Calculate average loss and Dice score for the epoch
            avg_train_loss = epoch_loss / len(self.train_loader.dataset)
            avg_train_dice = torch.mean(torch.stack(dice_scores)).item()
            avg_train_hd = torch.mean(torch.stack(hd_scores)).item()
            avg_train_iou = torch.mean(torch.stack(iou_scores)).item()
            std_train_dice = torch.std(torch.stack(dice_scores)).item()

            print(f'Epoch {epoch} Train loss {avg_train_loss:.4f} '
                    f'Train Dice {avg_train_dice:.4f} '
                    f'Train HD {avg_train_hd:.4f} '
                    f'Train IoU {avg_train_iou:.4f} '
                    f'Train std: {std_train_dice:.4f}')

            if epoch % 100 == 0:
                # Save model weights and plot loss graphs
                epoch_folder = os.path.join('output', f'epoch_{epoch}')
                os.makedirs(epoch_folder, exist_ok=True)

                model_weights_path = os.path.join(epoch_folder, 'model.pth')
                torch.save(model.state_dict(), model_weights_path)

                train_losses_path = os.path.join(epoch_folder, 'train_losses.png')
                self.plot_list(epoch_loss_list, title='Training Loss in each epoch',
                            xlabel='Epoch', ylabel='Loss Value',
                            out_path=train_losses_path)

            # Validation step
            if epoch % self.val_interval == 0:
                model.eval()
                val_epoch_loss = 0
                best_val_dice = 0.0
                best_epoch = 0
                val_dice_scores = []
                val_hd_scores = []
                val_iou_scores = []
                for step, data_val in enumerate(self.val_loader):
                    images = data_val["ct"].to(self.device)
                    seg = data_val["seg"].to(self.device)
                    batch_size = images.size(0)
                    with torch.no_grad():
                        with autocast(enabled=True):
                            prediction = model(images)
                            prediction = F.interpolate(prediction, size=seg.shape[2:], mode='trilinear', align_corners=False)
                            prediction_dice = torch.sigmoid(prediction)
                            prediction_dice = (prediction_dice > 0.5).float()
                            dice = DiceCoefficient(prediction_dice, seg)
                            hd = hausdorff_distance(seg, prediction_dice)
                            iou = iou_score(seg, prediction_dice)
                            
                            val_dice_scores.append(dice.cpu())
                            val_hd_scores.append(hd)
                            val_iou_scores.append(iou)
                            loss = combined_loss_function(prediction, seg)
                    
                    val_epoch_loss += loss.item() * batch_size
                    
                avg_val_loss = val_epoch_loss / len(self.val_loader)
                avg_val_dice = torch.mean(torch.stack(val_dice_scores)).item()
                avg_val_hd = torch.mean(torch.stack(val_hd_scores)).item()
                avg_val_iou = torch.mean(torch.stack(val_iou_scores)).item()
                std_val_dice = torch.std(torch.stack(val_dice_scores)).item()
                
                print("Epoch", epoch, 
                        "Validation loss", avg_val_loss, 
                        "Val dice", avg_val_dice, 
                        "Val HD", avg_val_hd,
                        "Val IoU", avg_val_iou,
                        "Val std", std_val_dice)
                val_epoch_loss_list.append(avg_val_loss)
                val_dice_score_list.append(avg_val_dice)

                if avg_val_dice > best_val_dice:
                    best_val_dice = avg_val_dice
                    best_epoch = epoch
                    self.save_checkpoint(model, optimizer, epoch, avg_val_loss)
                
                    print(f"Best Epoch: {best_epoch} with Validation Dice Score: {best_val_dice:.4f}")    
                
                # # Plot validation metrics
                # val_losses_path = os.path.join(epoch_folder, 'val_losses.png')
                # self.plot_list(
                #             value_list=val_epoch_loss_list,
                #             title='Validation loss in each epoch',
                #             xlabel='Epoch',
                #             ylabel='Loss Value',
                #             out_path=val_losses_path
                #             )

                # dice_scores_path = os.path.join(epoch_folder, 'dice_scores.png')
                # self.plot_list(
                #             value_list=val_dice_score_list,
                #             title='Dice score in each epoch',
                #             xlabel='Epoch',
                #             ylabel='Dice Score',
                #             out_path=dice_scores_path
                #             )   

                # Testing step
                model.eval()
                test_epoch_loss = 0
                test_dice_scores = []
                test_hd_scores = []
                test_iou_scores = []
                for step, data_test in enumerate(self.test_loader):
                    images = data_test["ct"].to(self.device)
                    seg = data_test["seg"].to(self.device)
                    batch_size = images.size(0)
                    
                    with torch.no_grad():
                        with autocast(enabled=True):
                            prediction = model(images)
                            prediction = F.interpolate(prediction, size=seg.shape[2:], mode='trilinear', align_corners=False)
                            prediction_dice = torch.sigmoid(prediction)
                            prediction_dice = (prediction_dice > 0.5).float()
                            dice = DiceCoefficient(prediction_dice, seg)
                            hd = hausdorff_distance(seg, prediction_dice)
                            iou = iou_score(seg, prediction_dice)
                            
                            test_dice_scores.append(dice.cpu())
                            test_hd_scores.append(hd)
                            test_iou_scores.append(iou)
                            test_loss = combined_loss_function(prediction, seg)
                    
                    test_epoch_loss += test_loss.item() * batch_size

                avg_test_loss = test_epoch_loss / len(self.test_loader)
                avg_test_dice = torch.mean(torch.stack(test_dice_scores)).item()
                avg_test_hd = torch.mean(torch.stack(test_hd_scores)).item()
                avg_test_iou = torch.mean(torch.stack(test_iou_scores)).item()
                std_test_dice = torch.std(torch.stack(test_dice_scores)).item()

                print(f'Test loss {avg_test_loss:.4f} '
                        f'Test Dice {avg_test_dice:.4f} '
                        f'Test HD {avg_test_hd:.4f} '
                        f'Test IoU {avg_test_iou:.4f} '
                        f'Test std {std_test_dice:.4f}')

                test_dice_score_list.append(avg_test_dice)
                
                # # Plot test metrics
                # test_losses_path = os.path.join(epoch_folder, 'test_losses.png')
                # self.plot_list(
                #     value_list=[avg_test_loss],
                #     title='Test loss',
                #     xlabel='Epoch',
                #     ylabel='Loss Value',
                #     out_path=test_losses_path
                # )

                # test_dice_scores_path = os.path.join(epoch_folder, 'test_dice_scores.png')
                # self.plot_list(
                #     value_list=test_dice_score_list,
                #     title='Test Dice score in each epoch',
                #     xlabel='Epoch',
                #     ylabel='Dice Score',
                #     out_path=test_dice_scores_path
                # )

        model.eval()
        test_epoch_loss = 0
        test_dice_scores = []

        for step, data_test in enumerate(self.test_loader):
            images = data_test["ct"].to(self.device)
            seg = data_test["seg"].to(self.device)
            batch_size = images.size(0)
            
            with torch.no_grad():
                with autocast(enabled=True):
                    prediction = model(images)
                    prediction = F.interpolate(prediction, size=seg.shape[2:], mode='trilinear', align_corners=False)
                    prediction_dice = torch.sigmoid(prediction)
                    prediction_dice = (prediction_dice > 0.5).float()
                    dice = DiceCoefficient(prediction_dice, seg)
                    test_dice_scores.append(dice.cpu())
                    test_loss = combined_loss_function(prediction, seg)
                
            test_epoch_loss += test_loss.item() * batch_size

        avg_test_loss = test_epoch_loss / len(self.test_loader)
        avg_test_dice = np.mean(test_dice_scores)
        std_test_dice = np.std(test_dice_scores)

        print("Final Test loss:", avg_test_loss, "Final Test dice:", avg_test_dice.item(), "Final Test std:", std_test_dice.item())

        # test_losses_path = os.path.join(epoch_folder, 'final_test_losses.png')
        # self.plot_list(
        #     value_list=[avg_test_loss],
        #     title='Final Test loss',
        #     xlabel='Epoch',
        #     ylabel='Loss Value',
        #     out_path=test_losses_path
        # )

        # test_dice_scores_path = os.path.join(epoch_folder, 'final_test_dice_scores.png')
        # self.plot_list(
        #     value_list=[avg_test_dice],
        #     title='Final Test Dice score',
        #     xlabel='Epoch',
        #     ylabel='Dice Score',
        #     out_path=test_dice_scores_path
        # )

        checkpoint_last = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'last.pth'))
        torch.save(model.state_dict(), checkpoint_last)

        total_time = time.time() - total_start
        print(f"Training model completed, total time: {total_time}.")
