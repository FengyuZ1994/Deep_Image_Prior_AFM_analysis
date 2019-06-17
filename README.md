## Deep Image Prior in Morphology Data Analysis

Atomic force microscopy (AFM) is an effective tool to study morphology and texture of diverse surfaces. Atomic force microscopic analysis is ideal for quantitatively measuring the nano-metric dimensional surface roughness and for visualizing the surface nano-texture of the deposited film. However, AFM data always contains various types of artifacts such as probe artifacts, scanner artifacts, vibrations and random noises. For example, the unavoidable experimental errors, such as “tip crash,” the large height variation of the sample surface, and the interaction between the tip and the ambient environment, can severely reduce the spatial resolution of the AFM images. Several methods and techniques have been proposed to enhance the resolution and quality of the AFM images, such as the modification of the tip, the development of the contour metrology, and the combination with other microscopic techniques. 

![Gołek, F., et al. "AFM image artifacts." _Applied Surface Science_ 304 (2014): 11-19.](https://raw.githubusercontent.com/FengyuZ1994/test_imgs/master/AFM%20nosie%20demo.png)


In addition to the development of new experimental techniques, data processing is important before viewing, analyzing and enhancing AFM images. Most AFM products are supplied with powerful analysis and image display software. Among all SPM data analysis tools, Gwyddion is the most powerful software which provides a large number of [data processing functions](http://gwyddion.net/features.php#processing), including all the standard statistical characterization, leveling and data correction, filtering or grain marking functions. However, Gwyddion has its own limitations when a degraded figure or a low-resolution figure is used as an input. For example, scars (or stripes, strokes) are parts of the image that are corrupted by a very common scanning error: local fault of the closed loop due to sudden change in-depth profile. Images with scars or outliers usually have very low contrast and the small details in it will be barely visible. Gwyddion solves this problem by masking the artifacts area and reconstruct the affected area by interpolating with outer boundary pixels. However, this method tends to give an unsatisfactory approximation of the unaffected area, especially when the mask is much larger than the size of a single pixel. Another common issue of SPM is its low resolution, usually 512 px x 512 px, due to the size of the tip and long scanning time. The zoomed-in images of grains, textures, and patterns have even worse resolutions, but Gwyddion does not support state of the art resolution enhancement modules.

In recent years, owing to the dramatic increase in the computational capabilities, deep learning (DL) techniques have been implemented in a variety of research fields. For image processing, DL, especially the deep convolutional neural network (CNN), provides powerful and effective tools to perform various image processing tasks such as image denoise, image segmentation, image resolution enhancements, etc. Most of those deep neural network methods only work in the context of massive datasets or models pre-trained on such datasets. This might not be a problem in applications of natural image processing since there is a massive dataset of natural images that have contributed greatly to the advancement in this field. However, the lack of massive and reliable dataset of AFM images has limited the implementation of most state-of-the-art DL techniques in AFM image analysis.

Recently, a startling paper, [“Deep Image Prior”](http://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html), showing that the **structure** of the CNN contains “pre-knowledge” of natural images. Deep neural networks are powerful enough to memorize a single image. This means that theoretically, it would seem likely that the CNN would just reproduce the original noisy image. It showed that some tasks such as denoising and super-resolution can actually be successfully conducted on a **single image**, **without any additional training data**. This is an extremely inspiring result because it not only shows that some some deep neural networks could be successfully trained on a single image, but also the structure of the network itself could be preventing deep networks from overfitting. The most interesting part of this paper though: in practice, when we optimize using gradient descent, CNNs “resist” noisy images or images with artifacts, and has a bias towards producing natural (low-noise) images. The following figure provides evidence to support this claim, showing that the loss converges much faster for natural images compared to noise. This means that if we cut off the training at an appropriate timing or step, one can obtain a “natural” image. This is why [this paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html) regards CNN as a prior: it **has a bias towards producing natural images**.

![enter image description here](https://i0.wp.com/mlexplained.com/wp-content/uploads/2018/01/learning_curves.png?w=774)


# Applications to AFM data analysis

There are many applications of the CNN as a prior. Different use cases require slightly different loss functions as shown in [this paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html), but the underlying principles are the same.

## 1. Denoising and general reconstruction
The original Deep Image Prior paper demonstrates the applicability of this method to a wide range of denoising applications. A variety of noises,  such as salt-pepper noisy, Gaussian noise and white noise, are found to be removed successfully using this method. The following figure shows an example: 

![enter image description here](https://i1.wp.com/mlexplained.com/wp-content/uploads/2018/01/denoising_example.png?w=1076)

Now, Deep Image Prior is applied to denoise the real AFM image with synthetic salt&pepper noise. As shown in the figure below, the top left shows the ground true low-noise image and the top right shows the corrupted image with synthetic noisy which is used as the only input to deep CNN prior. After 1000 iterations, the synthetic noisy has been removed reasonably from the input image. Yet after 3000 iterations, the CNN starts to learn the salt&pepper noise from the input image. This example shows not only the effectiveness of Deep Image Prior in removing noisy from AFM images but also the necessity of choosing a stop iteration number to avoid fitting the noisy. 

![enter image description here](https://raw.githubusercontent.com/FengyuZ1994/test_imgs/master/de13.png)

## 2. **Super resolution**

The goal of super-resolution is to take a low-resolution image and upsample it to create a high-resolution version. This is indeed the feature that people need for AFM data analysis. The low-resolution (corrupted) image is used as input. 

![enter image description here](https://raw.githubusercontent.com/FengyuZ1994/test_imgs/master/SR.001.png)
![](https://raw.githubusercontent.com/FengyuZ1994/test_imgs/master/SR.002.png)
![enter image description here](https://raw.githubusercontent.com/FengyuZ1994/test_imgs/master/SR1.png)


## 3. Inpainting

Inpainting is a task where some of the pixels in an image are replaced with a blank mask, and the erased portion has to be reconstructed. The figure below shows an example of using Deep Image Prior for image inpainting. The pixels under the blank mask are reconstructed perfectly, which [outperforms some complicated sparse coding based approaches](http://mlexplained.com/2018/01/18/paper-dissected-deep-image-prior-explained/). 
![enter image description here](https://i0.wp.com/mlexplained.com/wp-content/uploads/2018/01/inpainting_example.png?w=1056)

An AFM image usually contains line noises or square noises when the vibrating tip "hit" sharp peaks or clusters on the surface. The figure below shows those two common noises present in AFM images. Unfortunately, the current interpolation method used in AFM analysis has been giving unsatisfactory results. The corrupted areas in the image are strongly affected by the surrounding pixels. Therefore, it is interesting to test the performance of Deep Image Prior on those AFM images. 

![enter image description here](https://raw.githubusercontent.com/FengyuZ1994/test_imgs/master/line&square%20noises.png.001.png)

We starting the evaluation by applying CNN image prior to some ground truth images with different simulated masks. 

![enter image description here](https://raw.githubusercontent.com/FengyuZ1994/test_imgs/master/Inpainting1.png)
![](https://raw.githubusercontent.com/FengyuZ1994/test_imgs/master/Inpainting2.png)
![enter image description here](https://raw.githubusercontent.com/FengyuZ1994/test_imgs/master/Inpainting3.png)

Those images above demonstrate the power of CNNs as priors over AFM images. The reconstructed images are almost identical to the ground truth image, regardless of the type of masks. We also found that during inpainting, even when large portions of the input were masked out, the deep image prior managed to take the surrounding texture and fill the holes appropriately. Based on the results, I proposed that the AFM corrupted by line or square noises can be processed in the following way: 
#### 1. Remove the background in AFM image.
#### 2. Mask the corrupted areas with masks (line, square, circle, etc.)
#### 3. Randomly generate weights for CNN and pass the masked image as input.
#### 4. Manually choose the iteration numbers to avoid the overfitting of CNN to the noises or artifacts. 
This procedure worked for various of AFM images, giving very "real" reconstructed images, despite it has its own limitations. If an AFM has irregular features, such as one white dot in a black background and the dot is corrupted, the CNN prior certainly can not "guess" the pixels covered under that mask. But in general, this method has been giving quite satisfactory results when choosing appropriate learning parameters.  



# Conclusion and further reading

Deep image prior is an intriguing paper that has widespread implications for the field of computer vision. This method can be applied to not only natural images but also to images from Scanning Probe Microscopies such as AFM. It enables multiple image processing functions, such as denosing, super resolution, and inpainting, without training or pre-training on massive datasets. The inpainting by deep image prior is particularly useful, when combined with masks, to analyze corrupted AFM images, which enables fast and effective AFM image reconstructions. 

