# awesome-photorealistic-style-transfer
A survey of photorealistic style transfer (PST), including paper and code.

Style transfer is a task to recompose the content of an image in the style of another. After transfer, the ouput image has the content of content image but with the style of style image. This technology can be used for art creation and photo retouching. Different from artistic style transfer, this repo mainly refers to PST, which indicates that the result is visully realistic. Although this repo mainly focuses on photorealistic style transfer (PST), it contains some general paper of neural style transfer for basic understanding.

## Review paper
### *[TVCG2020]* Neural Style Transfer: A review [[paper](https://arxiv.org/abs/1705.04058)] [[code](https://github.com/ycjing/Neural-Style-Transfer-Papers)]  
comment:This paper provides a comprehensive overview of the current progress towards neural style transfer (NST). This paper propose a taxonomy of NST techniques. According to the classification, most of the papers in this repo belongs to model-optimization-based offline arbitrary-style-per-model photorealistic NST methods.

## Papers listed by year
### 2016
#### [CVPR2016 artistic] Image Style Transfer Using Convolutional Neural Networks [[paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)]  
comment: This paper is a seminar work of using CNN to render a content image with a style defined by the style image. This paper proposed a image-optimization-based online method. Style consistency is supported by feature correlation (Gram matrix).
#### *[CVPR2016 artistic]* Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [[paper](https://ieeexplore.ieee.org/document/7780634)]  
comment: Adaptive Instance Normalization (AdaIN) as a style transfer. The Instance Normalization parameter is derived from style image feature.

### 2017
#### [BMVC2017 artistic] Exploring the structure of a real-time, arbitrary neural artistic stylization network [[paper](https://arxiv.org/abs/1705.06830v2)]  
comment: Use Inceptionv3 to predict the AdaIN parameters.

#### [BMVC2017] Photorealistic Style Transfer with Screened Poisson Equation [[paper](https://arxiv.org/abs/1709.09828)]  
comment: Propose a post-processing method to stylized image via Screened Poisson Equation.

#### [Computer Society2017] Deep photo style transfer [[paper](https://arxiv.org/pdf/1703.07511.pdf)]  
comment: Key contribution is Photorealism regularization and augmented style loss with semantic segmentation.


### 2018
#### *[ECCV2018]* A Closed-form Solution to Photorealistic Image Stylization [[paper](https://arxiv.org/abs/1802.06474)] [[code](https://github.com/NVIDIA/FastPhotoStyle)]  
comment: *Recommand to reimplement.* A two stage method which includes a stylization step and a smoothing step. Both steps have a closed-form solution (can be computed eﬃciently). The stylization step is based on the whitening and coloring transform (WCT), which stylizes images via feature projections. The smoothing step is to optimize a smoothness term and a ﬁtting term which leads to a quadratic problem. Knowledge about 
linear algebra is need.

#### [ACM MM 2018] Structure Guided Photorealistic Style Transfer [[paper](https://dl.acm.org/doi/10.1145/3240508.3240637)]  
comment: A novel patch matching algorithm which simultaneously takes high-level category information and geometric structure information (e.g., human pose and building structure) into account.

### 2019
#### *[ICCV2019]* Photorealistic Style Transfer via Wavelet Transforms [[paper](https://arxiv.org/abs/1903.09760)][[code](https://github.com/clovaai/WCT2)]  
comment: *Recommand to reimplement.* Existing methods are limited by spatial distortions or unrealistic artifacts. This paper proposes a wavelet corrected transfer (WCT2) based on whitening and coloring transforms that allows features to preserve their structural information and statistical properties of VGG feature space during stylization.

[ACML2019] High-Resolution Network for Photorealistic Style Transfer [[paper](https://arxiv.org/abs/1904.11617)][[code](https://github.com/limingcv/Photorealistic-Style-Transfer)]  
comment: This method uses high-resolution generation network for better performance. 



### 2020
[ECCV 2020] Joint Bilateral Learning for Real-time Universal
Photorealistic Style Transfer [[paper](https://arxiv.org/pdf/2004.10955.pdf)][[code](https://github.com/mousecpn/Joint-Bilateral-Learning)]  
comment: *Recommand to reimplement.* The method is perform on *Bilateral Space*, thus it is edge-preserve and fast! The transformation is global smooth.

[AAAI2020] Ultrafast Photorealistic Style Transfer via Neural Architecture Search [[paper](https://arxiv.org/abs/1912.02398)][[code](https://github.com/pkuanjie/StyleNAS)]  
comment: adopt a neural architecture search method to accelerate PhotoNet. An automatic network pruning framework in the manner of teacher-student learning for photorealistic stylization. Ultrafast(achieving 20-30 times acceleration).

### 2021
[ICCV2021] Domain-Aware Universal Style Transfer [[paper](https://arxiv.org/pdf/2108.04441v2.pdf)][[code](https://github.com/Kibeom-Hong/Domain-Aware-Style-Transfer)]  
comment: This method supports for both artistic and photorealistic style transfer. To this end, they design a novel domainness indicator that captures the domainness value from the texture and structural features of reference images. Moreover, they introduce a unified framework with domainaware skip connection to adaptively transfer the stroke and palette to the input contents guided by the domainness indicator.

[TIP2021] Efficient Style-Corpus Constrained Learning for Photorealistic Style Transfer [[paper](https://ieeexplore.ieee.org/document/9360460)]  
comment: The style-corpus with the style-specific and style-agnostic characteristics simultaneously is proposed to constrain the stylized image with the style consistency among different samples, which improves photorealism of stylization output. By using adversarial distillation learning strategy, a simple fast-to-execute network is trained to substitute previous complex feature transforms models, which reduces the computational cost significantly (13 ~ 50 times faster than STOA models).
