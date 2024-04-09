import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import openslide
import cv2
import numpy as np
from PIL import Image
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from PIL import Image, ImageOps
import shutil
import glob
ON_GPU = True
from tqdm import tqdm
import joblib
import torch
from filters.util import pil_to_np_rgb, np_to_pil
from filters.filter import apply_image_filters
import math
import multiprocessing
import torch.nn.functional as F
from config import getConfig
# Our TLS segmentation model
from lib.PSCANet_ab import PSCANet
from torchvision import transforms

from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.architecture.hovernet import HoVerNet

# Use the TLS pretrained model and set the default data augmentation setting. 
opt = getConfig()
TLS_net = PSCANet(opt).cuda()
seg_model_path = './pretrained_model/16_best_model.pth'
TLS_net.load_state_dict(torch.load(seg_model_path))
TLS_net.cuda()
TLS_net.eval()

CropSize = 352
img_transforms = transforms.Compose([
        transforms.Resize((CropSize, CropSize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])]
        )


# initialize HoVerNet, referred from: https://github.com/TissueImageAnalytics/tiatoolbox
lymphocyte_model = HoVerNet(num_types=5, mode='fast')
pretrained = torch.load('./pretrained_model/hovernet_fast-monusac.pth')
lymphocyte_model.load_state_dict(pretrained)
lymphocyte_model = lymphocyte_model.cuda()
lymphocyte_model.eval()

def merge_patches(res_canvas, patch, start_x, start_y, stride):
    patch_size = patch.shape[1]
    end_x = start_x + patch_size
    end_y = start_y + patch_size
    res_canvas[start_y:end_y, start_x:end_x] = np.bitwise_or(res_canvas[start_y:end_y, start_x:end_x], patch)


def get_lyms_nums(original_img, model):
    width, height = original_img.size

    patch_size = 164
    stride = patch_size // 2  # 50% overlap
    padded_size = 256  # 模型需要的输入尺寸

    num_patches_x = math.ceil((width - stride) / stride)
    num_patches_y = math.ceil((height - stride) / stride)

    res_canvas = np.zeros((height, width), dtype=np.uint8)

    for i in range(num_patches_x):
        for j in range(num_patches_y):
            x = i * stride
            y = j * stride
            x = min(x, width - patch_size)
            y = min(y, height - patch_size)
            img_patch = original_img.crop((x, y, x + patch_size, y + patch_size))
            processed_patch = process_patch(img_patch, model)
            merge_patches(res_canvas, processed_patch, x, y, stride)
    return res_canvas

def process_patch(patch, model):
    # expanding patch with size of 164x164 to 256x256.
    padding = (256 - 164) // 2
    padded_patch = ImageOps.expand(patch, border=padding, fill='black')
    batch = torch.from_numpy(np.array(padded_patch))[None].cuda()
    with torch.no_grad():
        output = model.infer_batch(model, batch, on_gpu=True)
        output = [v[0] for v in output]
        output = model.postproc(output)
        infos = output[1]
        mask = np.zeros((164, 164), np.uint8)
        keys = infos.keys()
        for k in keys:
            # the type of 2 is the lymphocyte. 
            if infos[k]['type'] == 2:
                polypoints = infos[k]['contour']
                cv2.drawContours(mask, [polypoints], -1, (255), -1)
    return mask


save_dir = './save_img_results'
deal_img = './64_35.png'
lymphocyte_number = 80
#ExtractSize = 1024
# We cropped the patch with 1024x1024 based on the 40x magnification. For the 20x magnification, we could cropped the patch with 512x512 and resized the size to 1024x1024 for segmenting lymphocyte. When the lymphocyte number was bigger than 80, we restored the TLS segmentation and lymphocyte cell prediction results. However, these experimental parameters' settings were just our recommendation. 

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(os.path.join(save_dir, 'lyms')):
    os.makedirs(os.path.join(save_dir, 'lyms'))

if not os.path.exists(os.path.join(save_dir, 'tls')):
    os.makedirs(os.path.join(save_dir, 'tls'))

img_patch = Image.open(deal_img).convert('RGB')
ExtractSize = img_patch.size[0]

lymphocyte_patch = img_patch.resize((1024, 1024), Image.ANTIALIAS)
res_canvas = get_lyms_nums(lymphocyte_patch, lymphocyte_model)
res_image = Image.fromarray(res_canvas.astype('uint8'), 'L')
mask_img = np.array(res_image)
mask_img[mask_img >= 127] = 255
mask_img[mask_img < 127] = 0

# 检查mask_img中是否至少有一个像素是255（白色）
if np.sum(mask_img == 255) > 0:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_img)
    if num_labels >= lymphocyte_number:    
        img_tensor = torch.unsqueeze(img_transforms(img_patch), 0)
        inputs = img_tensor.cuda()
        P1, P2, P3, P4, P5 = TLS_net(inputs)
        res = F.upsample(P1, size=(ExtractSize, ExtractSize), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        TLS = Image.fromarray(res*255).convert('RGB')
        lymphocyte_img = Image.fromarray(mask_img.astype('uint8'), 'L')
        TLS.save(os.path.join(save_dir, 'tls', deal_img.split('/')[-1]))
        lymphocyte_img.save(os.path.join(save_dir, 'lyms', deal_img.split('/')[-1]))
else:
    print("Check your image, maybe this image dont have TLS area!")
    #pass


