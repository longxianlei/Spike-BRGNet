# Spike-BRGNet: Efficient and Accurate Event-based Semantic Segmentation with Boundary Region-Guided Spiking Neural Networks

This is the official repository for our recent work: Spike-BRGNet ([PDF](https://ieeexplore.ieee.org/document/10750266)）


* **Event only**:  The spike-based computing mechanism of Spike-BRGNet allows it to deal with complex scenarios efficiently and accurately without image frames.
* **A Novel Boundary-Guided Spiking Semantic Segmentation Model**: A three-branch SNN that consists of a semantic detail (SD), context aggregation(CA), and boundary region recognition (BA) branch, respectively.
* **More Accurate and Less Energy Cost**: Spike-BRGNet outperforms SOTA methods by at least 1.57% and 1.91% mIoU on DDD17 and DSEC-Semantic datasets with less computation cost, respectively.

## Updates
   - This paper was accepted by TCSVT 2024, new version and associated materials will be available soon! (Nov/8/2024)
   - Our paper was submitted to IEEE Explore for public access. (Nov/11/2024)
   - The training and testing codes and trained models for Spike-BRGNet are available here. (Dec/09/2024)



## Models
The finetuned models on DDD17 and DSEC-Semantic are available for direct application in road scene parsing.

| Model (DDD17) | Val (% mIOU) |
|:-:|:-:|
| Spike-BRGNet | [54.72](/home/ubuntu/code/SpikeBRGNet/output/DDD17_events/pidnet_small_DDD17/best.pt) 

| Model (DSEC) | Val (% mIOU) |
|:-:|:-:|
| Spike-BRGNet | [54.95](/home/ubuntu/code/SpikeBRGNet/output/DSEC_events/pidnet_small_DSEC/best.pt)

## Prerequisites
The inference speed is tested on single RTX 3090 using the method introduced by [SwiftNet](https://arxiv.org/pdf/1903.08469.pdf). No third-party acceleration lib is used, so you can try [TensorRT](https://github.com/NVIDIA/TensorRT) or other approaches for faster speed.

## Usage

### 0. Prepare the dataset
# DDD17 Dataset
The original DDD17 dataset with semantic segmentation labels can be downloaded here[https://github.com/Shathe/Ev-SegNet]. Additionally, the pre-processed DDD17 dataset with semantic labels is provided here[https://download.ifi.uzh.ch/rpg/ESS/ddd17_seg.tar.gz]. Please do not forget to cite DDD17 and Ev-SegNet if you are using the DDD17 with semantic labels.

# DSEC-Semantic Dataset
The DSEC-Semantic dataset can be downloaded here[https://dsec.ifi.uzh.ch/dsec-semantic/]. The dataset should have the following format:

├── DSEC_Semantic                 
│   ├── train               
│   │   ├── zurich_city_00_a  
│   │   │   ├── semantic  
│   │   │   │   ├── left
│   │   │   │   │   ├── 11classes
│   │   │   │   │   │   └──data
│   │   │   │   │   │       ├── 000000.png
│   │   │   │   │   │       └── ...
│   │   │   │   │   └── 19classes
│   │   │   │   │       └──data
│   │   │   │   │           ├── 000000.png
│   │   │   │   │           └── ...
│   │   │   │   └── timestamps.txt
│   │   │   └── events  
│   │   │       └── left
│   │   │           ├── events.h5
│   │   │           └── rectify_map.h5
│   │   └── ...
│   └── test
│       ├── zurich_city_13_a
│       │   └── ...
│       └── ... 

* Remenber to replace the dataset path in the yaml with your ture dataset path


### 1. Training
* For example, train the SpikeBRGNet-s on DDD17 with batch size of 32 on 1 GPUs:
````bash
python tools/train.py --cfg configs/DDD17/SpikeBRGNet_small_DDD17.yaml TRAIN.BATCH_SIZE_PER_GPU 32
````
* Or train the SpikeBRGNet-s on DSEC with batch size of 12 on 1 GPUs:
````bash
python tools/train.py --cfg configs/DSEC/SpikeBRGNet_small_DSEC.yaml TRAIN.BATCH_SIZE_PER_GPU 12
````

### 2. Evaluation

* Put the finetuned models for DDD17 and DSEC-Semantic into `output/DDD17_event/SpikeBRGNet_x_DDD17/` and `output/DSEC_events/SpikeBRGNet_x_DSEC/` dirs, respectively.
* For example, evaluate the SpikeBRGNet-S on DDD17 val set:
````bash
python tools/eval.py --cfg configs/DDD17/SpikeBRGNet_small_DDD17.yaml
````


## Citation

If you think this implementation is useful for your work, please cite our paper:
```
@ARTICLE{10750266,
  author={Long, Xianlei and Zhu, Xiaxin and Guo, Fangming and Chen, Chao and Zhu, Xiangwei and Gu, Fuqiang and Yuan, Songyu and Zhang, Chunlong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Spike-BRGNet: Efficient and Accurate Event-based Semantic Segmentation with Boundary Region-Guided Spiking Neural Networks}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Cameras;Semantic segmentation;Semantics;Computational modeling;Event detection;Data models;Circuits and systems;Spiking neural networks;Neurons;Robot vision systems;Boundary region guidance;Event camera;Event-based semantic segmentation;Spiking neural network;Traffic scenes segmentation},
  doi={10.1109/TCSVT.2024.3495769}}

```

## Acknowledgement

* Our implementation is modified based on [PIDNet](https://github.com/XuJiacong/PIDNet).
* Latency measurement code is borrowed from the [DDRNet](https://github.com/ydhongHIT/DDRNet).
* Thanks for their nice contribution.

