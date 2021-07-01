# FDA: Fourier Domain Adaptation for Semantic Segmentation

In this article, I will review the paper [FDA: Fourier Domain Adaptation for Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf) which is published in CVPR 2020. After an introduction to the topic, I will explain the method proposed in the paper and its contributions.

## Introduction
### Unsupervised Domain Adaptation (UDA)
Over the last decade, deep neural networks have achieved impressive success in various computer vision tasks, in presence of large amount of data. However, if we test a trained model on different data, it is very likely that the performance of the model drops significantly. As an example, we can think of a situation where we have a model trained on synthetic images and target data is real world images. In this case, we expect that the model will not provide the same success on this dataset. In order to avoid poor results, domain adaptation should be performed.

In unsupervised domain adaptation setting, we have labeled data from source domain and unlabeled data from target domain. Our aim is to adapt the model trained on source data to use it on target data with minimized performance drop. 

### UDA for Semantic Segmentation
To train a network for semantic segmentation, we need plenty of data and it is a demanding task to obtain this amount of data and label it. Hence, synthetic data can be used to train the network and we can use this network for segmentation of real images. However, this will lead to performance drop and we need to perform domain adaptation to mitigate this issue. The paper proposes a new method for this task.

### Notation
In this section, I will introduce the notation defined in paper and use it thorughout this article. 
<img width="180" alt="Ekran Resmi 2021-06-30 23 54 05" src="https://user-images.githubusercontent.com/56236171/124036639-9afb1980-d9fe-11eb-911a-b9a117892a5d.png"> represents source domain dataset, where <img width="120" alt="Ekran Resmi 2021-07-01 00 30 47" src="https://user-images.githubusercontent.com/56236171/124039644-a270f180-da03-11eb-9ff2-40930a537e5c.png"> is an RGB image and <img width="100" alt="Ekran Resmi 2021-07-01 00 32 41" src="https://user-images.githubusercontent.com/56236171/124039775-e237d900-da03-11eb-9761-3731839ab240.png"> is the corresponding ground truth semantic map. Similarly, <img width="90" alt="Ekran Resmi 2021-07-01 00 43 20" src="https://user-images.githubusercontent.com/56236171/124040543-62ab0980-da05-11eb-8dff-2b8fdbc2a2f4.png"> respresent target domain dataset, where <img width="120" alt="Ekran Resmi 2021-07-01 00 30 47" src="https://user-images.githubusercontent.com/56236171/124040986-4e1b4100-da06-11eb-88b0-9036b3c54118.png"> is an image and we do not have ground truth maps for this set. 

### Datasets
In the paper, two different synthetic datasets are used as source domain data separately. GTA5[1] has 24,966 annotated images with spatial resolution 1914×1052 originally, which resized to 1280×720 and randomly cropped to the 1024×512 in training process. The second synthetic dataset is SYNTHIA[2], its SYNTHIA-RAND-CITYSCAPES subset is used in paper which consists of 9,400 annotated images with size 1280×760. The images are randomly cropped to 1024×512 for training. 
As real-world dataset, CityScapes[3] is chosen. 2,975 images from this dataset is used as target domain data and 500 validation images are used for testing. These images are resized to 1024×512. 

## Related Work

## Method and Main contributions
### Spectral Transfer
In the paper, a spectral tansfer block is proposed. The reason behind implementation of the such a transfer is that variation of low-level spectrum does not affect high level semantics. However, neural networks learn these statistics together with useful features about semantics. To eliminate this, spectral transfer block applies the following steps:
- The first step is taking Fourier transform of randomly sampled target image and source image. You can see the formula here. The transform has an amplitude and a phase part. After taking the transform, they replace low frequency part of the amplitude of the source image with that of the target image. They define a mask with parameter beta. This parameter determines the size 	of the part to be replaced. In the end, taking the inverse Fourier transform gives them target stylized source image. The image will have the same semantic map with the original source image but its appearance will be similar to target images.


