# NeRFrac

<img src="https://github.com/Yifever20002/NeRFrac/blob/main/images/real_fish.gif" alt="fish" width="320" height="240"> | <img src="https://github.com/Yifever20002/NeRFrac/blob/main/images/real_redflower.gif" alt="redflower" width="320" height="240">
<img src="https://github.com/Yifever20002/NeRFrac/blob/main/images/real_tree.gif" alt="tree" width="320" height="240">
<img src="https://github.com/Yifever20002/NeRFrac/blob/main/images/real_plant.gif" alt="plant" width="320" height="240">


This is the repo for the ICCV 2023 project

>[NeRFrac: Neural Radiance Fields through Refractive Surface](https://openaccess.thecvf.com/content/ICCV2023/html/Zhan_NeRFrac_Neural_Radiance_Fields_through_Refractive_Surface_ICCV_2023_paper.html)\
>Yifan Zhan, Shohei Nobuhara, Ko Nishino, Yinqiang Zheng\
>IEEE International Conference on Computer Vision (ICCV), 2023

![image](https://github.com/Yifever20002/NeRFrac/blob/main/images/pipeline.png)

## About NeRFrac

Neural Radiance Fields (NeRF) is a popular neural expression for novel view synthesis. By querying spatial points and view directions, a multilayer perceptron (MLP) can be trained to output the volume density and radiance at each point, which lets us render novel views of the scene. The original NeRF and its recent variants, however, target opaque scenes dominated by diffuse reflection surfaces and cannot handle complex refractive surfaces well. We introduce NeRFrac to realize neural novel view synthesis of scenes captured through refractive surfaces, typically water surfaces. For each queried ray, an MLP-based Refractive Field is trained to estimate the distance from the ray origin to the refractive surface. A refracted ray at each intersection point is then computed by Snell's Law, given the input ray and the approximated local normal. Points of the scene are sampled along the refracted ray and are sent to a Radiance Field for further radiance estimation. We show that from a sparse set of images, our model achieves accurate novel view synthesis of the scene underneath the refractive surface and simultaneously reconstructs the refractive surface. We evaluate the effectiveness of our method with synthetic and real scenes seen through water surfaces. Experimental results demonstrate the accuracy of NeRFrac for modeling scenes seen through wavy refractive surfaces.


## Installation

``````
git clone https://github.com/Yifever20002/NeRFrac.git
cd NeRFrac
conda create -n NeRFrac python=3.6
conda activate NeRFrac
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install -r requirements.txt
``````


## Demo Data

This part encompasses single-frame multi-view data in both the main text and the supplementary material, including images and [COLMAP](https://colmap.github.io/)-calibrated camera poses (with original image resolution, 2048&times;1536 for real data and 392&times;392 for synthetic data). We use "images_4" for real data training and "images_1" for synthetic data training. Download from [here](https://drive.google.com/drive/folders/1A78v0qNCQlqS01AD77IqjhNrL9p0rkBF?usp=sharing) and place them under /pth/to/NeRFrac/data/nerf_llff_data/ .


## How to run?

To train a specific data, use
``````
python run_nerfrac.py --config configs/syn_primary_sine.txt
``````

To render a pre-trained model, set render_only == 1 and render_test == 1.

To visualize the training process, use
``````
tensorboard --logdir=./tensorboard --port 8123
``````

## Citation
``````
@inproceedings{zhan2023nerfrac,
  title={Nerfrac: Neural radiance fields through refractive surface},
  author={Zhan, Yifan and Nobuhara, Shohei and Nishino, Ko and Zheng, Yinqiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18402--18412},
  year={2023}
}
``````



