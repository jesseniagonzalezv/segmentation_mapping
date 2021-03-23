import helper
from collections import defaultdict
from helper import reverse_transform
from torch.utils.data import DataLoader
from loss import dice_loss, metric_jaccard, levelsetLoss, gradientLoss2d
from dataset import ImagesDataset
import torch.nn.functional as F
from models import UNet11, UNet, AlbuNet34, SegNet
import numpy as np
import torch
import glob
import os
import numpy as np
from pathlib import Path
from scalarmeanstd import meanstd
import pdb

from transformsdata import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        Rotate,
                        CenterCrop,
                        VerticalFlip)


def calc_loss(pred, target, metrics,phase='train', bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)

    # convering tensor to numpy to remove from the computationl graph 
    if  phase=='test':
        pred=(pred >0.50).float()  #with 0.55 is a little better
        dice = dice_loss(pred, target)
        jaccard_loss = metric_jaccard(pred, target)    
        loss = bce * bce_weight + dice[0] * (1 - bce_weight)
            
        metrics['bce'] = bce.data.cpu().numpy() * target.size(0)
        metrics['loss'] = loss.data.cpu().numpy() * target.size(0)
        metrics['dice'] = 1 - dice[0].data.cpu().numpy() * target.size(0)
        metrics['dice_class'] = 1 - dice[1].data.cpu().numpy() * target.size(0)
        metrics['jaccard'] = 1 - jaccard_loss[0].data.cpu().numpy() * target.size(0)
        metrics['jaccard_class'] = 1 - jaccard_loss[1].data.cpu().numpy() * target.size(0)
        
    else:
        dice = dice_loss(pred, target)
        jaccard_loss = metric_jaccard(pred, target)    
        loss = bce * bce_weight + dice[0] * (1 - bce_weight)
        
        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        metrics['dice_loss'] += dice[0].data.cpu().numpy() * target.size(0)
        metrics['dice_class'] += dice[1].data.cpu().numpy() * target.size(0)
        metrics['jaccard_loss'] += jaccard_loss[0].data.cpu().numpy() * target.size(0)
        metrics['jaccard_class'] += jaccard_loss[1].data.cpu().numpy() * target.size(0)


    return loss

# Using semi-supervised approach with Mumford-Shah loss functional
def calc_loss_MS(pred, target, inputs):
    
    criterionCE = torch.nn.CrossEntropyLoss()
    criterionLS = levelsetLoss()
    criterionTV = gradientLoss2d()
        
    loss_C = 0
    numch = 0
    
    #Supervised training method
    #targetCE = torch.argmax(target,dim=1)
    #if torch.max(targetCE[0, 0]) != 0: # if target exists
    #    loss_C = criterionCE(pred,targetCE)
       
    #Calculates cross entropy per image, checks if image has target label
    for ibatch in range(target.shape[0]):
        if torch.max(target[ibatch, 0]) != 0: # if target exists
            
            realB = target[ibatch, :].unsqueeze(0)
            realB = torch.argmax(realB, dim=1)
            fakeB = pred[ibatch, :].unsqueeze(0)
           
            loss_C += criterionCE(fakeB, realB.long())  # * 100
            numch += 1.0
    
    if numch > 0:
        loss_C = loss_C / numch
    else:
        loss_C = 0
        
    m = torch.nn.Softmax(dim=1)
    predSoftmax =  m(pred)
    #predClamps = torch.clamp(pred[:,:], 1e-10, 1.0)

    loss_L = criterionLS(predSoftmax, inputs)
    loss_A = criterionTV(predSoftmax) * 0.001
    loss_LS = (loss_L + loss_A) * 0.000001 #lambda_A (weight for cycle loss (A -> B -> A))

    loss_tot = loss_C + loss_LS

    return loss_tot


def print_metrics(metrics, file, phase='train', epoch_samples=1 ):    
    outputs = []
    for k in metrics.keys():
        if k == 'dice_class':
            outputs.append('dice_class: {:4}'.format('{}'.format(metrics[k] / epoch_samples)))
        elif k == 'jaccard_class':
            outputs.append('jaccard_class: {:4}'.format('{}'.format(metrics[k] / epoch_samples)))
        else:
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples ))
    if phase=='test':
        file.write("{}".format(",".join(outputs)))
    else:          
        print("{}: {}".format(phase, ", ".join(outputs)))
        file.write("{}: {}".format(phase, ", ".join(outputs)))    ### f

 
    
def make_loader(file_names,channels, shuffle=False, transform=None,mode='train',batch_size=1, limit=None):
        return DataLoader(
            dataset=ImagesDataset(file_names,channels, transform=transform,mode=mode, limit=limit),
            shuffle=shuffle,            
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available() 
        )

def find_metrics(train_file_names,val_file_names, test_file_names, channels,max_values, mean_values, std_values,model,fold_out='0', fold_in='0',  name_model='UNet11', epochs='40',out_file='VHR',dataset_file='VHR' ,name_file='_VHR_60_fake' ):
                            
    outfile_path = ('predictions/{}/').format(out_file)
        
    f = open(("predictions/{}/metric{}_{}_foldout{}_foldin{}_{}epochs.txt").format(out_file,name_file,name_model,fold_out, fold_in,epochs), "w+")
    f2 = open(("predictions/{}/pred_loss_test{}_{}_foldout{}_foldin{}_{}epochs.txt").format(out_file,name_file,name_model, fold_out, fold_in,epochs), "w+")
    f.write("Training mean_values:[{}], std_values:[{}] \n".format(mean_values, std_values))

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    #####Dilenames ###############################################

   
    print(len(test_file_names))
    #####Dataloder ###############################################

    all_transform = DualCompose([
                CenterCrop(int(dataset_file)),
                ImageOnly(Normalize(mean=mean_values, std=std_values))
            ])


    #train_loader = make_loader(train_file_names,channels,shuffle=True, transform=all_transform)
    #val_loader = make_loader(val_file_names,channels, transform=all_transform)
    test_loader = make_loader(test_file_names,channels, transform=all_transform)

    dataloaders = {'test':test_loader}

    for phase in ['test']:
        model.eval()
        metrics = defaultdict(float)
    ###############################  train images ###############################

        count_img=0
        input_vec= []
        labels_vec = []
        pred_vec = []
        result_dice = []
        result_dice_classes = []
        result_jaccard = []
        result_jaccard_classes = []
        
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)              
            with torch.set_grad_enabled(False):
                input_vec.append(inputs.data.cpu().numpy())
                labels_vec.append(labels.data.cpu().numpy())
                pred = model(inputs)

                loss = calc_loss(pred, labels, metrics,'test')
                
                if phase=='test':
                    print_metrics(metrics,f2, 'test')
                   
                pred=torch.sigmoid(pred)    
                pred_vec.append(pred.data.cpu().numpy())    

                result_dice += [metrics['dice']]
                
                result_dice_classes += [metrics['dice_class']]

                result_jaccard += [metrics['jaccard']]
                
                result_jaccard_classes += [metrics['jaccard_class']]

                count_img += 1

        print(("{}_{}").format(phase,out_file))
        print('Dice = ', np.mean(result_dice), np.std(result_dice))
        print('Dice classes= ', np.mean(result_dice_classes,axis=0))
        print('Jaccard = ',np.mean(result_jaccard), np.std(result_jaccard))
        print('Jaccard classes= ', np.mean(result_jaccard_classes,axis=0),'\n')

        f.write(("{}_{}\n").format(phase,out_file))
        f.write("dice_metric: {:4f}, std: {:4f} \n".format(np.mean(result_dice),np.std(result_dice)))
        f.write("dice_per_class: {:4} \n".format('{}'.format(np.mean(result_dice_classes,axis=0))))
        f.write("jaccard_metric: {:4f}, std: {:4f}  \n".format(np.mean(result_jaccard), np.std(result_jaccard)))    
        f.write("jaccard_per_class: {:4} \n".format('{}'.format(np.mean(result_jaccard_classes,axis=0))))
    
        if phase=='test':      
            np.save(str(os.path.join(outfile_path,"inputs_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file,name_model,fold_out,fold_in,epochs,int(count_img)))), np.array(input_vec))
            np.save(str(os.path.join(outfile_path,"labels_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file,name_model, fold_out,fold_in,epochs,int(count_img)))), np.array(labels_vec))
            np.save(str(os.path.join(outfile_path,"pred_test{}_{}_foldout{}_foldin{}_{}epochs_{}.npy".format(name_file,name_model, fold_out,fold_in,epochs,int(count_img)))), np.array(pred_vec))



