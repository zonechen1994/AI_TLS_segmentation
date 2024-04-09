# AI_TLS_segmentation

In this study, we use a deep learning model to automatically segment the TLS (Tertiary Lymphoid Structures) regions across various cancer types. For example:

![image](https://github.com/zonechen1994/AI_TLS_segmentation/assets/47493620/718c87da-6e25-44e0-9512-a9a69b4944eb)

As demonstrated, the trained deep learning (DL) model is capable of segmenting Tertiary Lymphoid Structures (TLS) from Hematoxylin and Eosin (H&E) stained images. The multiplex Immunohistochemistry (mIHC) images serve as the ground truth for TLS. 

In our study, we trained the TLS segmentation model using two types of cancer: Esophageal Squamous Cell Carcinoma (ESCC) and Non-Small Cell Lung Cancer (NSCLC). Validation of our model was performed using data from TCGA-ESCC (a subset of TCGA-ESCA) and TCGA-NSCLC (comprising TCGA-LUAD and TCGA-LUSC datasets). Finally, to evaluate the model's generalizability, we tested it against 14 additional pancancer cohorts from the TCGA database."

㊗️ **Our study has been accepted by npj Precision Oncology**! 🍺👏 

## 1. Enviroment setup

You can use the `requirements.txt` file to install all the necessary Python packages on your server. You will need at least one NVIDIA GPU.

<code>pip install -r requirements.txt </code>

Alternatively, you can use Conda to install all the essential packages.



### 2. The Segmentation architecture

The model architecture is located in  <code>lib/PSCANet_ab.py</code>. We used  EF-B0 as the backbone, and we can modify it with different backbones to validate prediction performance. 

The model for segmenting TLS reuses four modules from another article of ours, titled " <a href="https://arxiv.org/abs/2309.08234" title="Polyp segmentation model">Polyp segmentation model</a>" , which was accepted by IEEE ICASSP 2024. 



#### 2.1 Train TLS or polyp segmentation model. 

You can see the detail in the <code>Train.py</code>.  You can modify  the corresponding dataset patch. 

#### 2.2 Test TLS or polyp segmentation model in H&E tiles 

You can see the detail in the <code>Test.py</code>



## 3.TLS segmentation

#### 3.1 Predict TLS regions combined with the lymphocyte number 

👀 **We have released our TLS segmentation model now !!!!!**

You can download the corresponding pretrained models by Baidu Netdisk [Pretrained model](https://pan.baidu.com/s/10w58utK-n9MMhayTSGVi4w) and the extracted code is **"ptgf"** or you can download by Google driver [Pretrained model](https://drive.google.com/drive/folders/12i30PYvQayrc-HPN3J-1Dnd0iyisL17-?usp=sharing). Then, you need to put these two model checkpoints into the "pretrained_model" folder.  


As demonstrated in some previous studies, the TLS region needs to contain more than a certain number of lymphocytes. In this paper, we used the HoVer-Net from the Tiatoolbox to segment and quantify the number of lymphocytes.

We take the <code>64_35.png</code> as an example. By running <code>python get_patch_tls.py</code>, we obtain the results of TLS segmentation and lymphocyte segmentation regions in the folder of <code>save_img_results</code>. The pretrained models for HoVer-Net and TLS segmentation can be found in the <code>pretrained_model</code> folder. 

The corresponding parameters can been seen in the <code>get_patch_tls.py</code> file. You can adjust the parameters based on your results. Based this <code>get_patch_tls.py</code> file, you can predict all TLS regions in H&E slide by cropping patches.  



#### 3.2 Test TLS segmentation model in H&E slide

You can first download TCGA slides and then use our model to segment TLS areas. To download H&E slides for TCGA cancer cohorts, you can utilize the `gdc-client` software.

In our project, we use TCGA-HNSC as an example. After downloading all HNSC slides and storing them in an HNSC directory, we can calculate the TLS ratio as follows:"

<code>python get_slide_tls.py --cancer_type HNSC</code>

⚠️ **Caution:** You need to convert your files into a format that can be recognized by the `openslide` package. We store the results for each patch. Depending on your specific needs, you can modify the detailed code. 

For converting the format of H&E slide, we used the 'Pathomation' python package. You can follow my Bilibili account to see the tutorial: [format convertting](https://www.bilibili.com/video/BV1x94y1N7uw/)

Initially, you should run the `get_patch_tls.py` script to know the results of our pipeline.

Afterward, you will be able to obtain the TLS area and tissue area, enabling you to quantify the TLS ratio for each patient. The tissue area was calculated based on the non-zero pixels after applying the OTSU method to the whole tissue.  You can also calculate the tissue for each patch and summarize the total area.



#### 3.3 Find the TLS ratio results of the TCGA cohorts

If you need to use the TLS ratio to support your research, you can contact me by sending the email: ziqiangchen21@m.fudan.edu.cn






### 4. Reference codebase

1. <a href="https://github.com/Karel911/TRACER/tree/main" title="TRACER">TRACER</a>
2. <a href="https://github.com/DengPingFan/Polyp-PVT" title="Polyp-PVT">Polyp-PVT</a>
3. <a href="https://github.com/deroneriksson/python-wsi-preprocessing" title="wsi-preprocessing">wsi-preprocessing</a>

### 5. Future works

1. Train Pan-cancer tumor segmentation models to identify the intra-tumor or peri-tumor TLS. 
2. Train TLS status models to identify the mature or immature TLS. 
3. build a 'whl' package to segment TLS. 

If you want to design some clinical research based on TLS, you can contact me also!



### 6. Welcome to refer our studies

```
@article{chen2024deep,
  title={Deep learning on tertiary lymphoid structures in hematoxylin-eosin predicts cancer prognosis and immunotherapy response},
  author={Chen, Ziqiang and Wang, Xiaobing and Jin, Zelin and Li, Bosen and Jiang, Dongxian and Wang, Yanqiu and Jiang, Mengping and Zhang, Dandan and Yuan, Pei and Zhao, Yahui and others},
  journal={NPJ Precision Oncology},
  volume={8},
  number={1},
  pages={73},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

```
@inproceedings{chen2024efficient,
  title={Efficient polyp segmentation via integrity learning},
  author={Chen, Ziqiang and Wang, Kang and Liu, Yun},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1826--1830},
  year={2024},
  organization={IEEE}
}
```



