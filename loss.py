import torch
import torch.nn as nn

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))   #1-Dice
    
    return loss.mean() #mean of the batch



def metric_jaccard(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()  
    epsilon= 1e-15  #epsilon! para evitar el indeterminado
    intersection = (pred*target).sum(dim=-2).sum(dim=-1)
    union = target.sum(dim=-2).sum(dim=-1) + pred.sum(dim=-2).sum(dim=-1) - intersection
    djaccard=(intersection +epsilon ) / (union + epsilon)
    return djaccard.mean()


def metric_dice(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()  
    epsilon= 1e-15   
    intersection = (pred*target).sum(dim=-2).sum(dim=-1)
    ddice=2* intersection / (target.sum(dim=-2).sum(dim=-1)+ pred.sum(dim=-2).sum(dim=-1) + 1e-15)
    return ddice.mean()