import torch
import os
import argparse
from datetime import datetime
from lib.PSCANet_ab import PSCANet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import matplotlib.pyplot as plt
import random
from config import getConfig
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']='0'
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def validate(model, path, state):
    model.eval()
    imgs_lists = []
    gts_lists = []
    if state == 'validate':
        data_path = os.path.join(path, 'val.txt')
    elif state == 'test':
        data_path = os.path.join(path, 'test.txt')
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.strip().split(',')
            imgs_lists.append(os.path.join(path,content[0]))
            gts_lists.append(os.path.join(path,content[1]))
    num1 = len(imgs_lists)
    test_loader = test_dataset(imgs_lists, gts_lists, 352)
    DSC = 0.0
    valid_loss = 0
    for i in tqdm(range(num1)):
        with torch.no_grad():
            image, gt, name = test_loader.load_data()
            gt = gt.cuda()
            image = image.cuda()
            res, res1, res2, res3, res4 = model(image)
            loss_P1 = structure_loss(res, gt)
            loss_P2 = structure_loss(res1, gt)
            loss_P3 = structure_loss(res2, gt)
            loss_P4 = structure_loss(res3, gt)
            loss_P5 = structure_loss(res4, gt)
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4 + loss_P5
            valid_loss += loss
            gt = gt.cpu().numpy()
            gt /= (gt.max() + 1e-8)
       
            # eval Dice
            res = F.upsample(res , size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            input = res
            target = np.array(gt)
            N = gt.shape
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC = DSC + dice
    return valid_loss/ num1, DSC / num1



def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    global early_stop
    min_loss = 100000
    size_rates = [0.75, 1, 1.25] 
    loss_P1_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = images.cuda()
            gts = gts.cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2, P3, P4, P5 = model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            loss_P5 = structure_loss(P5, gts)
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4 + loss_P5
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P1_record.update(loss_P1.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P1_record.show()))
    # save model 
    save_path = (os.path.join(opt.train_save, opt.model))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #torch.save(model.state_dict(), save_path +str(epoch)+ 'PSCANet.pth')
    # choose the best model

    global dict_plot
   
    test1path = '../../datasets/TLS_Segmentation/TLS_data'
    if (epoch + 1) % 1 == 0:
        dice_list = [] 
        dataset_loss, meandice = validate(model, test1path, 'validate')
        logging.info('epoch: {}, dice: {}'.format(epoch, dataset_dice))
        dice_list.append(dataset_dice)
        #dict_plot[dataset].append(dataset_dice)
        
        if dataset_loss < min_loss:
            early_stop = 0
        else:
            early_stop += 1

         if early_stop == early_stopping:
            break
        
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path +'/{}_best_model.pth'.format(opt.refine_channels))
            print('##############################################################################best', best)
            logging.info('##############################################################################best:{}'.format(best))

        

    
if __name__ == '__main__':
    ##################model_name#############################
    model_name = 'PSCANet'
    ###############################################
    opt = getConfig() 
    logging.basicConfig(filename='{}_{}.log'.format(opt.model, opt.refine_channels),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    # seed constant
    seed = opt.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = PSCANet(opt).cuda()
    best = 0
    early_stop = 0
    early_stopping = 10
    params = model.parameters()
    #model = torch.nn.DataParallel(model, device_ids=[0,1])

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
        #optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    
    #image_root = '{}/images/'.format(opt.train_path)
    #gt_root = '{}/masks/'.format(opt.train_path)
    dir_doot = '../../datasets/TLS_Segmentation/TLS_data'
    imgs_list = []
    gts_list = []
    with open(os.path.join(dir_doot, 'train.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            contents = line.strip().split(',')
            imgs_list.append(os.path.join(dir_doot, contents[0]))
            gts_list.append(os.path.join(dir_doot, contents[1]))
    
    #train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
    #                          augmentation=opt.augmentation)
    train_loader = get_loader(imgs_list, gts_list, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        # you can adjust the lr, in this paper, we dont have adjust lr.  
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
    
