# Signal: Selective Interaction and Global-local Alignment for Multi-Modal Object Re-Identification
## News🍎
Our paper has been accepted to **AAAI-2026**🌹! Paper

## Environment🍊
``` 
conda create -n myenv python=3.10.13
conda activate myenv
cd {your path}
pip install -r requirements.txt
```
## Datasets🍋
* RGBNT201 [GET](https://drive.google.com/drive/folders/1EscBadX-wMAT56_It5lXY-S3-b5nK1wH)

* RGBNT100 [GET](https://pan.baidu.com/s/1xqqh7N4Lctm3RcUdskG0Ug?pwd=rjin)

* MSVR310 [GET](https://drive.google.com/file/d/1IxI-fGiluPO_Ies6YjDHeTEuVYhFdYwD/view)
## Pretrained Model🍉
* ViT-B-16 [GET](https://pan.baidu.com/share/init?surl=YPhaL0YgpI-TQ_pSzXHRKw) (52fu)
## Training🍒
```
python train.py --config_file configs/RGBNT201/Signal.yml
```
## Our Model🍇
Our model's pth file and training process record file are here: [Signal_model.pth](https://pan.baidu.com/s/1RUCXzp_EjsqOaPxWDssGsQ?pwd=sign)

## Test🥝
```
python test.py --config_file configs/RGBNT201/Signal.yml
```
## Introduction🧅️
To address multi-modal object ReID challenges, we propose **Signal**, a selective interaction and global-local alignment framework with three components:
* **Selective Interaction Module (SIM)**: Selects important patch tokens from multi-modal features via intra-modal and inter-modal attention scores.
* **Global Alignment Module (GAM)**: Simultaneously aligns multi-modal features by minimizing 3D polyhedra volume in gramian space.
* **Local Alignment Module (LAM)**: Refines fine-grained alignment via deformable sampling, handling pixel-level misalignment.
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

<p align="center">
    <img src="READ_image/Global.svg" alt="GAM" style="width:60%;">
</p>

<p align="center">
    <img src="READ_image/Local.svg" alt="LAM" style="width:100%;">
</p>
## Results🥂

<p align="center">
    <img src="READ_image/RGBNT201.png" alt="RGBNT201" style="width:100%;">
</p>

<p align="center">
    <img src="READ_image/RGBNT100_MSVR310.png" alt="RGBNT100_MSVR310" style="width:100%;">
</p>

<p align="center">
    <img src="READ_image/tokenvisual.svg" alt="tokenvisual" style="width:100%;">
</p>

<p align="center">
    <img src="READ_image/offsets.png" alt="offsets" style="width:100%;">
</p>

