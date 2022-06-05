import torch
import torch.nn as nn
import torch.nn.functional as F


'''base on: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss/focal_loss.py'''
class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss


def print_info(epoch, epochs, train_loss, train_acc, valid_loss, valid_acc):
    if epochs is not None:
        print(f'[ {epoch+1} ] | train_loss = {train_loss:.5f}, train_acc = {train_acc:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f}')
    else:
        print(f'[ {epoch+1}/{epochs} ] | train_loss = {train_loss:.5f}, train_acc = {train_acc:.5f}, val_loss = {valid_loss:.5f}, val_acc = {valid_acc:.5f}')