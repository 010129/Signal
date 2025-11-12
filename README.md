# Signal: Selective Interaction and Global-local Alignment for Multi-Modal Object Re-Identification
## News🍎
Our paper has been accepted to **AAAI-2026**🌹! Paper

## Environment🍊
``` 
conda create -n myenv python=3.10.13
conda activate myenv
pip install -r requirements.txt
```
## Datasets🍋
* RGBNT201 [GET](https://pan.baidu.com/s/1RUCXzp_EjsqOaPxWDssGsQ?pwd=sign)

* RGBNT100 [GET](https://pan.baidu.com/s/1RUCXzp_EjsqOaPxWDssGsQ?pwd=sign)

* MSVR310 [GET](https://pan.baidu.com/s/1RUCXzp_EjsqOaPxWDssGsQ?pwd=sign)
## Pretrained Model🍉
* ViT-B-16 [GET](https://pan.baidu.com/share/init?surl=YPhaL0YgpI-TQ_pSzXHRKw) (52fu)
## Training🍒
```
python train.py
```
## Our Model🍇
Our model's pth file and training process record file are here: [Signal_model.pth](https://pan.baidu.com/s/1RUCXzp_EjsqOaPxWDssGsQ?pwd=sign)

## Test🥝
```
python test.py
```
## Introduction🧅️


## Contributions🥬
* We propose a novel selective interaction and global-local alignment framework named Signal for multi-modal object ReID, which effectively addresses the challenges of background interference and multi-modal misalignment.
* We propose the Selective Interaction Module (SIM) to leverage inter-modal and intra-modal attention scores for selecting important patch tokens, thereby mitigating background interference in multi-modal fusion.
* We propose the Global Alignment Module (GAM) to  simultaneously align multi-modal features through minimizing the volume of 3D polyhedra in the gramian space.
* We propose the Local Alignment Module (LAM) to align local features in a shift-aware manner, effectively addressing pixel-level misalignment across modalities.
* Extensive experiments on three multi-modal object ReID datasets validate the effectiveness of our method.
## Overall Framework🍠
<p align="center">
    <img src="READ_image/main.svg" alt="Overall Framework" style="width:100%;">
</p>
## Results🥂

