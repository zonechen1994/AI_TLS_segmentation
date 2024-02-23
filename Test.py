import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ['CUDA_VISIBLE_DEVICES']='1'
from scipy import misc
from utils.dataloader import test_dataset
import cv2
from tqdm import tqdm
from lib.PSCANet_ab import PSCANet
from config import getConfig
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = getConfig()
    model = PSCANet(opt)
    model_path = (os.path.join(opt.train_save, opt.model, '{}_best_model.pth'.format(opt.refine_channels)))
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    #for _data_name in tqdm(['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']):
    ##### put data_path here #####
    data_path = '../../datasets/TLS_Segmentation/TLS_data/'

    
    stage = opt.stage
    if not os.path.exists(os.path.join('result_maps')):
        os.makedirs('result_maps')

    if not os.path.exists(os.path.join('result_maps', opt.model)):
        os.makedirs(os.path.join('result_maps', opt.model))

    if not os.path.exists(os.path.join('result_maps', opt.model, stage)):
        os.makedirs(os.path.join('result_maps', opt.model, stage))

    save_path = './result_maps/{}/{}/'.format(opt.model, stage)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    file_path = os.path.join(data_path, stage + '.txt')
    imgs = []
    gts = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            contents = line.split(',')
            imgs.append(contents[0])
            gts.append(contents[1])
    
    #image_root = '{}/images/'.format(data_path)
    #gt_root = '{}/masks/'.format(data_path)
    #num1 = len(os.listdir(gt_root))
    CropSize = opt.trainsize
    img_transforms = transforms.Compose([
        transforms.Resize((CropSize, CropSize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])]
        )
    with torch.no_grad():

        nums = len(imgs)
        for i in tqdm(range(nums)):
            img_path = imgs[i]
            dirs, name = img_path.split('/')
            if not os.path.exists(os.path.join(save_path, dirs)):
                os.makedirs(os.path.join(save_path, dirs))
            patch_path = os.path.join(save_path, dirs, name)

            img = Image.open(os.path.join(data_path, img_path)).convert('RGB')
            h, w = img.size
            img_tensor = torch.unsqueeze(img_transforms(img), 0)
            inputs = img_tensor.cuda()
        
            P1, P2, P3, P4, P5 = model(inputs) 
            res = F.upsample(P1, size=(h, w), mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            im = Image.fromarray(res*255).convert('RGB')
            im.save(patch_path)
        
        print('Finish!!!!!')
