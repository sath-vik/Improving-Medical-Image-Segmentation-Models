import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import albumentations as A

import archs

import losses
from dataset import Dataset
from lumbarSpineDataset import LumbarSpineDataset

from metrics import iou_score, indicators

from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter

import shutil
# import os # Already imported
import subprocess

from pdb import set_trace as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f"Using device : {device}")


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')

    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='ISIC', help='dataset name') #KEEP THIS.
    parser.add_argument('--data_dir', default='/kaggle/input/isic2017data', help='dataset dir')

    parser.add_argument('--output_dir', default='/kaggle/working/outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')



    config = parser.parse_args()

    return config

#Rest of the code remains the same.
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice, _ = iou_score(outputs[-1], target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(outputs[-1], target)

        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # Sets the seed for a single GPU
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[config['loss']]().to(device)

    cudnn.benchmark = True

    # create model
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'], no_kan=config['no_kan'])

    model = model.to(device)


    param_groups = []

    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)
        if 'layer' in name.lower() and 'fc' in name.lower(): # higher lr for kan layers
            # kan_fc_params.append(name)
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']})
        else:
            # other_params.append(name)
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})




    # st()
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)


    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
    shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')

    # --- DATASET LOADING (MODIFIED FOR MULTIPLE DATASETS) ---
    if config['dataset'] == 'ISIC':
        img_ext = '.jpg'
        mask_ext = '_segmentation.png'
        train_img_dir = os.path.join(config['data_dir'], 'ISIC-2017_Training_Data')
        train_mask_dir = os.path.join(config['data_dir'], 'ISIC-2017_Training_Part1_GroundTruth')
        val_img_dir = os.path.join(config['data_dir'], 'ISIC-2017_Validation_Data')
        val_mask_dir = os.path.join(config['data_dir'], 'ISIC-2017_Validation_Part1_GroundTruth')
        # Get image and mask IDs
        train_img_ids = sorted(glob(os.path.join(train_img_dir, '*' + img_ext)))
        train_mask_ids = sorted(glob(os.path.join(train_mask_dir, '*' + mask_ext)))
        val_img_ids = sorted(glob(os.path.join(val_img_dir, '*' + img_ext)))
        val_mask_ids = sorted(glob(os.path.join(val_mask_dir, '*' + mask_ext)))

    elif config['dataset'] == 'Fetus_Head':
        img_ext = '.png'
        mask_ext = '_Annotation.png'
        train_img_dir = os.path.join(config['data_dir'], 'training_set')
        train_mask_dir = os.path.join(config['data_dir'], 'training_set')
        val_img_dir = os.path.join(config['data_dir'], 'validation_set')
        val_mask_dir = os.path.join(config['data_dir'], 'validation_set')
        # Get image and mask IDs
        train_img_ids = sorted(glob(os.path.join(train_img_dir, '*' + img_ext)))
        train_mask_ids = sorted(glob(os.path.join(train_mask_dir, '*' + mask_ext)))
        val_img_ids = sorted(glob(os.path.join(val_img_dir, '*' + img_ext)))
        val_mask_ids = sorted(glob(os.path.join(val_mask_dir, '*' + mask_ext)))


    elif config['dataset'] == 'Lumbar_Spine':
        img_ext = '.png'
        mask_ext = '.png'  # Assuming mask extension is also .png
        # Define the base directory for the Lumbar Spine dataset
        base_dir = os.path.join(config['data_dir'], 'png') # Assuming 'png' is the folder.
        print("Base directory:", base_dir)

        # Create lists to store image and mask paths
        train_img_ids = []
        train_mask_ids = []
        val_img_ids = []
        val_mask_ids = []

        # Iterate through 'images' and 'masks' subdirectories
        image_dir = os.path.join(base_dir, 'images')
        mask_dir = os.path.join(base_dir, 'masks')
        #Checking paths:
        print(f"Checking image directory: {image_dir}")
        print(f"Checking mask directory: {mask_dir}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")


        for case_folder in sorted(os.listdir(image_dir)):  # Iterate through Case folders
            case_path_img = os.path.join(image_dir, case_folder)
            case_path_mask = os.path.join(mask_dir, case_folder) #Need mask path also
            if os.path.isdir(case_path_img) and os.path.isdir(case_path_mask):#Check for both

                image_files = sorted(glob(os.path.join(case_path_img, '*' + img_ext)))
                # print(image_files)

                for img_file in image_files:

                    img_filename = os.path.basename(img_file)

                    mask_filename = img_filename.replace('Slice', 'Labels_Slice')
                    mask_path = os.path.join(case_path_mask, mask_filename)
                    # print(mask_filename, mask_path)

                    if os.path.exists(mask_path):
                        # Now, do the 80/20 split:
                        if random.random() < 0.8:  # 80% for training
                            train_img_ids.append(img_file)
                            train_mask_ids.append(mask_path)
                        else:  # 20% for validation
                            val_img_ids.append(img_file)
                            val_mask_ids.append(mask_path)

                    else:
                        print(f"Warning: Mask not found for image: {img_file}") #Good practice.


        #Since we now have full paths, no need to define these:
        train_img_dir = ""
        train_mask_dir = ""
        val_img_dir = ""
        val_mask_dir = ""

    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    print("Current Working Directory:", os.getcwd()) #Added

    # Debugging prints (highly recommended!)
    print("Training image paths:", [os.path.join(train_img_dir, '*' + img_ext)])
    print("Number of training images:", len(train_img_ids))
    print("Number of training masks:", len(train_mask_ids)) # Check mask ids length
    print("Validation image paths:", [os.path.join(val_img_dir, '*' + img_ext)])
    print("Number of validation images:", len(val_img_ids))
    print("Number of validation masks:", len(val_mask_ids)) # Check mask ids length

    train_transform = Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),  # 50% chance of horizontal flip
        A.VerticalFlip(p=0.5),    # 50% chance of vertical flip
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),

    ])

    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])


    if config['dataset'] == 'Lumbar_Spine':
        train_dataset = LumbarSpineDataset(
            img_ids=train_img_ids,
            mask_ids=train_mask_ids,
            img_dir=train_img_dir,  # These will be empty strings, and that's OK
            mask_dir=train_mask_dir, # These will be empty strings
            transform=train_transform
        )
        val_dataset = LumbarSpineDataset(
            img_ids=val_img_ids,
            mask_ids=val_mask_ids,
            img_dir=val_img_dir,      # These will be empty strings
            mask_dir=val_mask_dir,    # These will be empty strings
            transform=val_transform
        )
    else:
        train_dataset = Dataset(
        img_ids=train_img_ids,
        mask_ids=train_mask_ids,  # This parameter is missing in train.py
        img_dir= train_img_dir,
        mask_dir= train_mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)

        val_dataset = Dataset(
            img_ids=val_img_ids,
            mask_ids=val_mask_ids,
            img_dir= val_img_dir,
            mask_dir= val_mask_dir,
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=config['num_classes'],
            transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])


    best_iou = 0
    best_dice= 0
    trigger = 0
    save_every_x_epochs = 10
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            trigger = 0


        if (epoch + 1) % save_every_x_epochs == 0:  # Check if it's a multiple of save_every_x_epochs
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model_epoch_{epoch+1}.pth')
            print(f"=> saved model at epoch {epoch+1}")

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()