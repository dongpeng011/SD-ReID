<div align="center">

# 🚀 SPDR-Net: Scene-Prompt-Driven Dynamic Routing Expert Network for Open-World Person Re-Identification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper: Under Review](https://img.shields.io/badge/Paper-Under_Review-orange.svg)]()
[![Dataset: SD-ReID](https://img.shields.io/badge/Dataset-SD--ReID-purple.svg)](https://github.com/dongpeng011/SD-ReID)

**Official PyTorch Implementation of the SPDR-Net**  
*Tackling Spatially Asymmetric Interference via Mixture of Experts (MoE) & Dynamic Routing*

<img src="assets/architecture.png" width="85%" alt="SPDR-Net Framework">
<br>
*(Figure: The overall architecture of SPDR-Net, integrating Semantic Part Experts, Prompt-Guided Dynamic Routing, and Global-Local Fusion.)*

</div>

---

## 📢 News
- **[🎉 2026/03]** The source code of **SPDR-Net** is officially released! 
- **[🔥 2026/04]** Our self-built challenging street-scene dataset **SD-ReID** is now publicly available!
- **[🚀 2026/05]** Supported YOLO-based closed-loop ReID deployment pipeline for real-world surveillance.

---

## ✨ Key Features
- 🧠 **Mixture of Experts (MoE) Architecture**: Innovatively treats pedestrian physical topologies (Head, Torso, Legs) as independent expert paths.
- 🚦 **Prompt-Guided Dynamic Routing (PGDR)**: Uses high-level scene semantics as a "soft logical switch" to dynamically cut off noise-polluted feature paths (e.g., severe partial occlusion).
- 📉 **Sparsity Routing Entropy Loss ($L_{ent}$)**: A novel information-theoretic constraint that forces the network to make decisive, highly confident routing decisions.
- 🏙️ **The SD-ReID Benchmark**: A brand new, highly challenging dataset captured from real-world street cameras, featuring extreme resolution variance, illumination shift, and dynamic occlusions.

---

## 🗂️ Datasets Preparation

We conduct extensive experiments on **8 mainstream benchmarks** and our **SD-ReID**. Please download the datasets and place them in your `data` directory.

| Dataset | Scenario | Download Link |
| :--- | :--- | :--- |
| **SD-ReID (Ours)** | **Real Street/Composite** | [Download Here](https://github.com/dongpeng011/SD-ReID) 🌟 |
| Market-1501 | Normal | [Kaggle Link](https://www.kaggle.com/datasets/pengcw1/market-1501/data) |
| MSMT17 | Normal / Multi-scene | [PKU Link](http://www.pkuvmc.com/dataset.html) |
| Occ-Duke | Severe Occlusion | [GitHub Link](https://github.com/lightas/Occluded-DukeMTMC-Dataset) |
| SYSU-mm01 | Infrared (Cross-modality) | [SYSU Link](https://www.isee-ai.cn/project/RGBIRReID.html) |
| Celeb-ReID | Clothing-Changing | [GitHub Link](https://github.com/Huang-3/Celeb-reID) |
| PRCC | Clothing-Changing | [SYSU Link](https://www.isee-ai.cn/%7Eyangqize/clothing.html) |
| MLR-CUHK03 | Low Resolution | [Baidu Disk](https://pan.baidu.com/s/1hMQZq0LAPhIl5RQ_EDDiFg) |

### 📂 Directory Structure
After downloading, please organize your dataset directory as follows and modify the root paths in the `./configs/` files:
```text
SPDR-Net/
├── data/
│   ├── SD-ReID/
│   │   ├── bounding_box_train/
│   │   ├── bounding_box_test/
│   │   └── query/
│   ├── market1501/
│   ├── msmt17/
│   └── ...
```
---
## 🛠️ Environment Setup
We recommend using Linux (Ubuntu 22.04), Python >= 3.8, and PyTorch >= 2.0 (CUDA 12.1).
Clone this repository and install the dependencies:
# Clone the repository
git clone https://github.com/dongpeng011/SD-ReID.git
cd SD-ReID

# Create a conda environment
conda create -n spdr python=3.8 -y
conda activate spdr

# Install PyTorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt
(Note: Our environment setup is heavily inspired by the excellent TransReID.)
---

---
## 🚀 Getting Started
### 1. Training SPDR-Net
To train the SPDR-Net model on the SD-ReID dataset, simply run:
```
python train.py --config configs/sd_reid.yml \
                MODEL.DEVICE_ID "0" \
                SOLVER.MAX_EPOCHS 60 \
                OUTPUT_DIR "./logs/sd_reid_spdr_net"
```
(Tip: You can easily switch to other datasets like configs/market1501.yml or configs/occ_duke.yml)

## 2. Evaluation
To evaluate a trained model, use the following command:
```
python test.py --config configs/sd_reid.yml \
               TEST.WEIGHT "./logs/sd_reid_spdr_net/best_model.pth" \
               MODEL.DEVICE_ID "0"
```
---

## 📊 Main Results
SPDR-Net sets new state-of-the-art performances across multiple challenging scenarios!
```
Dataset	mAP (%)	Rank-1 (%)	Config	Weight
SD-ReID (Ours)	99.0	99.9	sd_reid.yml	Google Drive
Occ-Duke	62.1	75.6	occ_duke.yml	Google Drive
MSMT17	42.3	66.4	msmt17.yml	Google Drive
PRCC	53.1	40.3	prcc.yml	Google Drive
```

## 📖 Citation
If you find this code, the SD-ReID dataset, or our ideas useful in your research, please consider citing our paper:
```
@article{SPDRNet2024,
  title={Scene-Prompt-Driven Dynamic Routing Expert Network for Open-World Person Re-Identification},
  author={Your Name and Co-authors},
  journal={Under Review},
  year={2024}
}
```

## 🤝 Acknowledgements
We would like to thank the open-source community, particularly the authors of TransReID and timm for their excellent codebases.