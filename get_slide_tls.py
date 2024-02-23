import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', '9000000000')

import openslide
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
import shutil
import glob
ON_GPU = True
from tqdm import tqdm
import joblib
import torch
from filters.util import pil_to_np_rgb, np_to_pil
from filters.filter import apply_image_filters
import math
import torch.nn.functional as F
from torchvision import transforms
from config import getConfig
from lib.PSCANet_ab import PSCANet

# initialization for TLS segmentation model
opt = getConfig()
seg_model = PSCANet(opt).cuda()
seg_model_path = './model_pth/PSCANet_efficientNet/efficientnet-b0/16_best_model.pth'
seg_model.load_state_dict(torch.load(seg_model_path))
seg_model.cuda()
seg_model.eval()

# initialization for HoVerNet
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.architecture.hovernet import HoVerNet
model = HoVerNet(num_types=5, mode='fast')
pretrained = torch.load('hovernet_fast-monusac.pth')
model.load_state_dict(pretrained)
model = model.cuda()
model.eval()


# settings 
CropSize = 352
img_transforms = transforms.Compose([
        transforms.Resize((CropSize, CropSize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])]
        )
ExtractSize = 1024 # can modify
RepetitionRate = 0 # can modify
lymphocyte_numbers = 80 # can modify

# using HoverNet to segment and count number of lymphocyte 
def get_lyms_nums(original_img):
    width, height, channels = original_img.shape
    num = math.ceil(width / 164)
    dis = (width - num * 164) // 2

    canvas = np.zeros((num * 164, num * 164, 3))
    res_canvas = np.zeros((num * 164, num * 164))
    canvas[62:62+1024, 62:62+1024, :] = original_img
    counts = 0
    for i in range(num):
        for j in range(num):
            img = canvas[i*164:(i+1)*164, j*164:(j+1)*164, :]
            patch_canvas = np.zeros((256, 256, 3))
            patch_canvas[46:210, 46:210, :] = img
            batch = torch.from_numpy(patch_canvas)[None].cuda()
            with torch.no_grad():
                output = model.infer_batch(model, batch, on_gpu=True)
                output = [v[0] for v in output]
                output = model.postproc(output)
                infos = output[1]
                mask = np.zeros((164, 164), np.uint8)
                keys = infos.keys()
                for k in keys:
                    if infos[k]['type'] == 2:
                        counts += 1
                        polypoints = infos[k]['contour']
                        cv2.drawContours(mask, [polypoints], -1, (255), -1)
                res_canvas[i*164:(i+1)*164, j*164:(j+1)*164] = mask
    res_canvas = res_canvas.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(res_canvas)
    res_canvas = cv2.resize(res_canvas, (width, height))
    return num_labels, res_canvas

if __name__ == '__main__':
    cancer_type = opt.cancer_type
    predict_dirs = '/home/users/zqchen/datasets/{}'.format(cancer_type)
    save_dirs = './predictions/{}_Predictions'.format(cancer_type)
    save_dirs_lists = os.listdir(save_dirs)
    dirs = os.listdir(predict_dirs)
    
    if not os.path.exists(save_dirs):
        os.makedirs(save_dirs)
    
    for d in dirs:
        if not os.path.exists(os.path.join('predictions/tissue_predictions/{}'.format(cancer_type))):
            os.makedirs(os.path.join('predictions/tissue_predictions/{}'.format(cancer_type)))
        writer = open('predictions/tissue_predictions/{}/{}.txt'.format(cancer_type, d), 'w')
        curr_d =  os.path.join(predict_dirs, d)
        ref_img_path = glob.glob(os.path.join(curr_d, "*svs"))[0]
        name = ref_img_path.split('/')[-2]

        if not os.path.exists(os.path.join(save_dirs, name, 'images')):
            os.makedirs(os.path.join(save_dirs, name, 'images'))

        if not os.path.exists(os.path.join(save_dirs, name, 'masks')):
            os.makedirs(os.path.join(save_dirs, name, 'masks'))

        if not os.path.exists(os.path.join(save_dirs, name, 'lyms')):
            os.makedirs(os.path.join(save_dirs, name, 'lyms'))

        slide = openslide.OpenSlide(ref_img_path)
        
        if slide.properties['aperio.AppMag'] == '40':
            ExtractSize = 1024
        elif slide.properties['aperio.AppMag'] == '20':
            ExtractSize = 512
        else:
            ExtractSize = 1024
        width, height = slide.dimensions
        with torch.no_grad():
            
            for j in tqdm(range(int((width - ExtractSize * RepetitionRate) / (ExtractSize * (1 - RepetitionRate))))):
                for i in range(int((height - ExtractSize * RepetitionRate) / (ExtractSize * (1 - RepetitionRate)))):
                    patch = slide.read_region((int(j * ExtractSize * (1 - RepetitionRate)), int(i * ExtractSize * (1 - RepetitionRate))), 0, (ExtractSize, ExtractSize))

                    img_patch = patch.convert('RGB')
                    process = pil_to_np_rgb(img_patch)
                    # slide_num is a default parameter. I dont modify this function. Just keep original implementation. 
                    process = apply_image_filters(process, slide_num=200, info=None, save=False, display=False)
                    rgb_sum = np.sum(process[:, :, 0] != 0)

                    if rgb_sum < ExtractSize*ExtractSize*0.1:
                        continue
                    else :
                        patch = np.array(img_patch)

                        counts, res_cells = get_lyms_nums(patch)
                        if counts >= lymphocyte_numbers:
                            lyms_path = os.path.join(save_dirs, name, 'lyms', '{}_{}.png'.format(j, i))
                            cv2.imwrite(lyms_path, res_cells)

                            img_path = os.path.join(save_dirs, name, 'images', '{}_{}.png'.format(j, i))
                            img_patch.save(img_path)

                            img_tensor = torch.unsqueeze(img_transforms(img_patch), 0)
                            inputs = img_tensor.cuda()
                            P1, P2, P3, P4, P5 = seg_model(inputs)
                            res = F.upsample(P1, size=(ExtractSize, ExtractSize), mode='bilinear', align_corners=False)
                            res = res.sigmoid().data.cpu().numpy().squeeze()
                            im = Image.fromarray(res*255).convert('RGB')
                            mask_path = os.path.join(save_dirs, name, 'masks', '{}_{}.png'.format(j, i))
                            im.save(mask_path)
        
        level = slide.level_count - 1
        tissue = slide.read_region((0, 0), level, (width, height))
        tissue = tissue.convert('RGB') 
        tissue = np.array(tissue)
        tissue = cv2.cvtColor(tissue, cv.COLOR_RGB2GRAY)
        ret, tissue = cv2.threshold(tissue, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        tissue_imgs = np.sum(tissue != 0)
        writer.write(d + '\t' + str(tissue_imgs))
