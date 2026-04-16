# Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos (CVPR 2026)

Xuankai Zhang, Junjin Xiao, Shangwei Huang, Wei-shi Zheng, and Qing Zhang*

<!-- project page and paper -->
<!-- [Project Page](https://se3bsplinegs.github.io/) &nbsp; [Paper](https://arxiv.org/abs/2510.10691)  -->
[Paper](https://arxiv.org/abs/2603.25058)

<!-- pageviews -->
<!-- <a href="https://info.flagcounter.com/dhPB"><img src="https://s01.flagcounter.com/mini/dhPB/bg_FFFFFF/txt_000000/border_FFFFFF/flags_0/" alt="Flag Counter" border="0"></a> -->

<!-- teaser -->
![curve](asset/teaser.svg)


Our method synthesizes high-quality novel views from monocular videos.


## Method Overview
![workflow](asset/overviewv1.svg)

<!-- Our method's overall workflow. Dotted arrows and dashed arrows describe the pipeline for modeling camera motion blur and modeling defocus blur, respectively at training time. Solid arrows show the process of rendering sharp images at the inference time. Please refer to the paper for more details. -->

## Todo
<!-- - [ ] ~~Release Paper, Example Code~~ -->
- [x] ~~Release Paper~~ 
- [x] ~~Release Code~~ 
- [ ] Clean Code

 ## Setup
###  1. Installation
```
git clone https://github.com/hhhddddddd/se3bsplinegs.git --recursive 
cd se3bsplinegs

conda create -n se3bsplinegs python=3.10
conda activate se3bsplinegs

# install pytorch
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# install dependencies
pip install -r requirements.txt
```
### 2. dataset and checkpoints
- Download the iPhone and NVIDIA data, obtain the RAFT and TAPNet checkpoints, and preprocess the data by following [MoSca](https://github.com/JiahuiLei/MoSca).

- Download the Zero123 checkpoint by following [DreamScene4D](https://github.com/dreamscene4d/dreamscene4d).


### 3. Train 

```
python mosca_reconstruct.py --cfg ./profile/iphone/iphone_fit.yaml --ws ./data/iphone/apple
```

## Acknowledgement

This repo is developed based on several amazing prior works:

- `Mosca`: https://github.com/JiahuiLei/MoSca
- `SoM`: https://github.com/vye16/shape-of-motion
- `SplineGS`: https://github.com/KAIST-VICLab/SplineGS
- `HiMoR`: https://github.com/pfnet-research/himor
- `MarbleGS`: https://github.com/coltonstearns/dynamic-gaussian-marbles
- `MoDec-GS`: https://github.com/skwak-kaist/MoDec-GS
- `DreamScene4D`: https://github.com/dreamscene4d/dreamscene4d


## BibTeX
```
@article{zhang2026learning,
  title={Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos},
  author={Zhang, Xuankai and Xiao, Junjin and Huang, Shangwei and Zheng, Wei-shi and Zhang, Qing},
  journal={arXiv preprint arXiv:2603.25058},
  year={2026}
}
```


 
