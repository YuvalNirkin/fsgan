## FSGAN - Official PyTorch Implementation
![Teaser](./docs/teaser.gif)
Example video face swapping: Barack Obama to Benjamin Netanyahu, Shinzo Abe to Theresa May, and Xi Jinping to 
Justin Trudeau.

This repository contains the source code for the video face swapping and face reenactment method described in the paper:
> **FSGAN: Subject Agnostic Face Swapping and Reenactment**  
> *International Conference on Computer Vision (ICCV), Seoul, Korea, 2019*  
> Yuval Nirkin, Yosi Keller, Tal Hassner  
> [Paper](https://arxiv.org/pdf/1908.05932.pdf) &nbsp; [Video](https://www.youtube.com/watch?v=BsITEVX6hkE)
>
> **Abstract:** *We present Face Swapping GAN (FSGAN) for face swapping and reenactment. Unlike previous work, FSGAN is subject agnostic and can be applied to pairs of faces without requiring training on those faces. To this end, we describe a number of technical contributions. We derive a novel recurrent neural network (RNN)–based approach for face reenactment which adjusts for both pose and expression variations and can be applied to a single image or a video sequence. For video sequences, we introduce continuous interpolation of the face views based on reenactment, Delaunay Triangulation, and barycentric coordinates. Occluded face regions are handled by a face completion network. Finally, we use a face blending network for seamless blending of the two faces while preserving target skin color and lighting conditions. This network uses a novel Poisson blending loss which combines Poisson optimization with perceptual loss. We compare our approach to existing state-of-the-art systems and show our results to be both qualitatively and quantitatively superior.*

## Important note
**THE METHODS PROVIDED IN THIS REPOSITORY ARE NOT TO BE USED FOR MALICIOUS OR INAPPROPRIATE USE CASES.**  
We release this code in order to help facilitate research of technical counter-measures for detecting this
kind of forgeries. Suppressing this kind of publications will not stop their development but will only make
it more difficult to detect them. 

Please note this is a work in progress, while we make every effort to improve the results of this method, not
every pair of faces can produce a high quality face swap.


## Requirements
- High-end NVIDIA GPUs with at least 11GB of DRAM.
- Either Linux or Windows. We recommend Linux for better performance.
- CUDA Toolkit 10.1, CUDNN 7.5, and the latest NVIDIA driver.
- Python 3.6+ and PyTorch 1.4.0+.

## Installation
- [Ubuntu installation guide](https://github.com/YuvalNirkin/fsgan/wiki/Ubuntu-Installation-Guide)
- [Windows installation guide](https://github.com/YuvalNirkin/fsgan/wiki/Windows-Installation-Guide)

For accessing FSGAN's pretrained models and auxiliary data, please fill out
[this form](https://docs.google.com/forms/d/e/1FAIpQLScyyNWoFvyaxxfyaPLnCIAxXgdxLEMwR9Sayjh3JpWseuYlOA/viewform?usp=sf_link).
We will then send you a link to FSGAN's shared directory and download script.

## Inference
- [Face swapping guide](https://github.com/YuvalNirkin/fsgan/wiki/Face-Swapping-Inference)
- [Face swapping Google Colab](inference/face_swapping.ipynb)
- [Paper models guide](https://github.com/YuvalNirkin/fsgan/wiki/Paper-Models-Inference)

## Training
- [Training V2](https://github.com/YuvalNirkin/fsgan/wiki/Training-V2)

## Citation
```
@inproceedings{nirkin2019fsgan,
  title={{FSGAN}: Subject agnostic face swapping and reenactment},
  author={Nirkin, Yuval and Keller, Yosi and Hassner, Tal},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={7184--7193},
  year={2019}
}

@inproceedings{nirkin2022fsganv2,
  title={{FSGANv2}: Improved Subject Agnostic Face Swapping and Reenactment},
  author={Nirkin, Yuval and Keller, Yosi and Hassner, Tal},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
