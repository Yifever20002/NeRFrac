# NeRFrac

This is the repo for the ICCV 2023 project

>NeRFrac: Neural Radiance Fields through Refractive Surface,\
>Yifan Zhan, Shohei Nobuhara, Ko Nishino, Yinqiang Zheng\
>IEEE International Conference on Computer Vision (ICCV), 2023

Code coming soon...
![image](https://github.com/Yifever20002/NeRFrac/blob/main/images/pipeline.png)
## About NeRFrac

Neural Radiance Fields (NeRF) is a popular neural expression for novel view synthesis. By querying spatial points and view directions, a multilayer perceptron (MLP) can be trained to output the volume density and radiance at each point, which lets us render novel views of the scene. The original NeRF and its recent variants, however, target opaque scenes dominated by diffuse reflection surfaces and cannot handle complex refractive surfaces well. We introduce NeRFrac to realize neural novel view synthesis of scenes captured through refractive surfaces, typically water surfaces. For each queried ray, an MLP-based Refractive Field is trained to estimate the distance from the ray origin to the refractive surface. A refracted ray at each intersection point is then computed by Snell's Law, given the input ray and the approximated local normal. Points of the scene are sampled along the refracted ray and are sent to a Radiance Field for further radiance estimation. We show that from a sparse set of images, our model achieves accurate novel view synthesis of the scene underneath the refractive surface and simultaneously reconstructs the refractive surface. We evaluate the effectiveness of our method with synthetic and real scenes seen through water surfaces. Experimental results demonstrate the accuracy of NeRFrac for modeling scenes seen through wavy refractive surfaces.

