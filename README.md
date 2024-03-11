# AI_TLS_segmentation
Using Deep learning to segment and quantify the TLS ratios across cancer types. For examples:
![image](https://github.com/zonechen1994/AI_TLS_segmentation/assets/47493620/718c87da-6e25-44e0-9512-a9a69b4944eb)


As you can see, we can use the trained DL model to segment TLS from the H&E images. The mIHC images show the labels of TLS. 

❤️ Now, we have submitted to npj Precison Oncology. After our paper has accepted, we will release the corresponding TLS segmentation checkpoint model. 


## 1. Enviroment setup
You  can use the requirements.txt to install the whole python packages in your server. You need to have one NVIDIA GPU at least. 

<code>pip install -r requirements.txt </code>

or, you can use the conda to install the whole packages. 


## 2.For TLS segmentation in H&E tiles
### The model architecture
The model architecture is in <code>lib/PSCANet_ab.py</code>. We used the EF-B0 as the backbone. We can modify different backbones. 
The model is the same of our another article for segmenting polop area. <a href="https://arxiv.org/abs/2309.08234" title="Polyp segmentation model">Polyp segmentation model</a>, which was accepted in IEEE ICASSP 2024. 


### 2.1 Training TLS segmentation model in H&E tiles.
You can see the detail in the <code> Train.py </code>. 

### 2.2 Test TLS segmentation model in H&E tiles 
You can see the detail in the <code>Test.py</code>

### 3. Test TLS segmentation model in slides
You can first download TCGA slides, and use our model to segment TLS areas. You can use the gdc-client software to download H&E slides for TCGA cancer cohorts. 

In our project, we use the TCGA-HNSC for example. After we download all HNSC slides and stored in a HNSC directory, we can calculate TLS ratio like: 

<code>python get_slide_tls.py --cancer_type HNSC</code>


### 4. Reference codebase
1. <a href="https://github.com/Karel911/TRACER/tree/main" title="TRACER">TRACER</a>
2. <a href="https://github.com/DengPingFan/Polyp-PVT" title="Polyp-PVT">Polyp-PVT</a>
3. <a href="https://github.com/deroneriksson/python-wsi-preprocessing" title="wsi-preprocessing">wsi-preprocessing</a>
