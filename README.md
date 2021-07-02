
 # FDA: Fourier Domain Adaptation for Semantic Segmentation
 
In this article, I will review the paper [FDA: Fourier Domain Adaptation for Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf) which is published in CVPR 2020. After an introduction to the topic, I will explain the method proposed in the paper and its contributions.

## Introduction
### Unsupervised Domain Adaptation (UDA)
Over the last decade, deep neural networks have achieved impressive success in various computer vision tasks, in presence of large amount of data. However, if we test a trained model on different data, it is very likely that the performance of the model drops significantly. As an example, we can think of a situation where we have a model trained on synthetic images and target data is real world images. In this case, we expect that the model will not provide the same results on this new dataset. In order to avoid these poor results, domain adaptation should be performed.

In unsupervised domain adaptation setting, we have labeled data from source domain and unlabeled data from target domain. Our aim is to adapt the model trained on source data to use it on target data with minimized performance drop. 

### UDA for Semantic Segmentation
To train a network for semantic segmentation, we need plenty of data. High quality labeling takes 60 minutes for an image from CamVid[1] dataset and 90 minutes for CityScapes[2]. Hence, it is a demanding task to obtain this amount of data and label it for real-world images. As a solution to this issue, segmentation network can be trained with synthetic data, where labeling an image takes around 7 seconds[3] and this network used for real-world images. However, this will lead to performance drop and we need to perform domain adaptation to mitigate it. The paper proposes a new method for this task.


### Notation
In this section, I will introduce the notation defined in the paper and use it throughout this article.  
<img width="200" alt="Ekran Resmi 2021-07-02 13 08 50" src="https://user-images.githubusercontent.com/56236171/124266314-19f36d80-db37-11eb-9dba-1ff73f830aa9.png">
 represents source domain dataset, where <img width="120" alt="Ekran Resmi 2021-07-01 00 30 47" src="https://user-images.githubusercontent.com/56236171/124039644-a270f180-da03-11eb-9ff2-40930a537e5c.png"> is an RGB image and <img width="100" alt="Ekran Resmi 2021-07-01 00 32 41" src="https://user-images.githubusercontent.com/56236171/124039775-e237d900-da03-11eb-9761-3731839ab240.png"> is the corresponding ground truth semantic map. Similarly, <img width="90" alt="Ekran Resmi 2021-07-01 00 43 20" src="https://user-images.githubusercontent.com/56236171/124040543-62ab0980-da05-11eb-8dff-2b8fdbc2a2f4.png"> represents target domain dataset, where <img width="120" alt="Ekran Resmi 2021-07-01 00 30 47" src="https://user-images.githubusercontent.com/56236171/124040986-4e1b4100-da06-11eb-88b0-9036b3c54118.png"> is an image and we do not have ground truth maps for this set. 

### Datasets
In the paper, two different synthetic datasets are used as source domain data in two scenarios. GTA5[3] has 24,966 annotated images with spatial resolution 1914×1052 originally, which resized to 1280×720 and randomly cropped to the 1024×512 in training process. The second synthetic dataset is SYNTHIA[4], its SYNTHIA-RAND-CITYSCAPES subset is used in paper which consists of 9,400 annotated images with size 1280×760. The images are randomly cropped to 1024×512 for training. 
As real-world dataset, CityScapes[2] is chosen. 2,975 images from this dataset is used as target domain data and 500 validation images are used for testing. These images are resized to 1024×512. 



## Related Work
### CyCADA: Cycle-Consistent Adversarial Domain Adaptation[5]
One of the state of the art methods in the field is CyCADA, which uses cycle consistent adversarial network to get translated images. They generate target stylized source images, then reconstruct source images from translated data and compute the loss to train the cycle in the figure. Also, they force the original and translated images to have the same semantics with semantic consistency loss. In addition to that, they employ image level and feature level loss functions to improve this alignment and they have a loss function for segmentation.
<div align="center">
  <img width="600" alt="Ekran Resmi 2021-07-02 03 29 58" src="https://user-images.githubusercontent.com/56236171/124207450-d886a200-dae5-11eb-8eb7-c6122ed869ec.png">
</div> 
<div align="center">
 Cycle-consistent adversarial adaptation 
</div> 

### Bidirectional Learning for Domain Adaptation of Semantic Segmentation[6]
BDL is the other state-of-the-art method which is in the field of UDA. In the method, there are two separated networks. The first one is image-to-image translation model and the second one is segmentation adaptation model. The networks are trained in both ways.  
In forward direction, the first network is trained with source and target data and translated images are obtained, which have the same semantic maps as the original source images. The second network will be trained with them in addition to target images and ground truth labels of source domain data.  
In backward direction, the motivation is to promote translation model using updated segmentation model. Authors of the paper aim to improve quality of translated images in this direction.
 <div align="center">
  <img width="900" alt="Ekran Resmi 2021-06-04 12 14 07" src="https://user-images.githubusercontent.com/56236171/124207607-269ba580-dae6-11eb-8b74-747f213dc3e2.png">

</div>
<div align="center">
  Network architecture of BDL
</div>   

## Method and Main contributions
In this section, I will explain the method proposed in the paper in detail and point out its main contributions.
### Spectral Transfer
In the paper, a spectral transfer block is proposed. The reason behind implementation of such a transfer is that variation of low-level spectrum does not affect high level semantics. However, neural networks learn these statistics together with useful features about semantics. To eliminate this, the structure in the figure follows a certain path.  
<div align="center"> <img width="650" alt="Ekran Resmi 2021-07-01 10 26 43" src="https://user-images.githubusercontent.com/56236171/124092236-ec36f780-da56-11eb-8e7f-94f2be11402b.png"> </div>  
 <div align="center">
  Style transfer block proposed in paper
</div>  
  
The first thing it does is taking Fourier transform of randomly sampled target image and source image with the following formula: 
<div align="center">
  <img width="400" alt="Ekran Resmi 2021-07-01 10 19 31" src="https://user-images.githubusercontent.com/56236171/124091231-e2f95b00-da55-11eb-9dec-8a14f050de6a.png">
</div> 
This transform has an amplitude and a phase part. After taking the transform, it replaces low frequency part of the amplitude of the source image with the same part of the amplitude of target image. For this process, following mask should be used:  
<div align="center">
  <img width="300" alt="Ekran Resmi 2021-07-01 11 11 38" src="https://user-images.githubusercontent.com/56236171/124098779-2905ed00-da5d-11eb-83ae-7c95aeae6563.png">
</div> 




In the mask, <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> defines the area to be replaced. This is the only parameter used in transfer part and it takes values between 0 and 1. The center of image is accepted as (0, 0) and H refers to height while W refers to width. This definition of mask makes choice of <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> independent of the size of the image. In the end, taking inverse Fourier Transform gives the source image transferred to target domain <img width="45" alt="Ekran Resmi 2021-07-01 12 09 46" src="https://user-images.githubusercontent.com/56236171/124107164-40e16f00-da65-11eb-9b15-97de2500c23c.png">. Overall process with randomly sampled source image <img width="65" alt="Ekran Resmi 2021-07-01 11 39 45" src="https://user-images.githubusercontent.com/56236171/124106151-3c688680-da64-11eb-9301-900b46436d42.png">
 and target image <img width="60" alt="Ekran Resmi 2021-07-01 11 40 01" src="https://user-images.githubusercontent.com/56236171/124106018-1e9b2180-da64-11eb-81b7-c4aae7f5c58f.png"> can be expressed with the following formula:  
<div align="center">
  <img width="420" alt="Ekran Resmi 2021-07-01 11 54 43" src="https://user-images.githubusercontent.com/56236171/124106592-a123e100-da64-11eb-9d94-a8c307aa6bd6.png">
</div>  

Output images <img width="40" alt="Ekran Resmi 2021-07-01 12 22 48" src="https://user-images.githubusercontent.com/56236171/124109121-1abcce80-da67-11eb-997d-4dd3566f57fe.png">, will have the same semantic map as <img width="20" alt="Ekran Resmi 2021-07-01 12 20 16" src="https://user-images.githubusercontent.com/56236171/124108794-c580bd00-da66-11eb-9abe-2793b93f6fb7.png"> but its appearance will be similar to that of target images.

#### Choice of <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png">
The only parameter to be chosen in the spectral transfer part is <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png">. From the mask definition, we know that as <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> approaches to 0, <img width="40" alt="Ekran Resmi 2021-07-01 12 22 48" src="https://user-images.githubusercontent.com/56236171/124109121-1abcce80-da67-11eb-997d-4dd3566f57fe.png"> will be more similar to <img width="20" alt="Ekran Resmi 2021-07-01 12 20 16" src="https://user-images.githubusercontent.com/56236171/124108794-c580bd00-da66-11eb-9abe-2793b93f6fb7.png"> and as <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> approaches to 1, <img width="40" alt="Ekran Resmi 2021-07-01 12 22 48" src="https://user-images.githubusercontent.com/56236171/124109121-1abcce80-da67-11eb-997d-4dd3566f57fe.png"> will approach to <img width="20" alt="Ekran Resmi 2021-07-01 12 45 05" src="https://user-images.githubusercontent.com/56236171/124112063-35447700-da6a-11eb-9c56-cdd1a4fa4e29.png">
. However, this also brings significant artifacts to the output image. In the paper, the effect of different choices of <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> is visualized as in the figure. 
<div align="center">
  <img width="860" alt="Ekran Resmi 2021-07-01 12 40 27" src="https://user-images.githubusercontent.com/56236171/124111490-a2a3d800-da69-11eb-9286-43ba2043bffb.png">
</div> 
<div align="center">
  Translated images for different <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> values
</div>  
 

### Semi-Supervised Training
After spectral transfer part, training process starts. Since the spectral transfer did not change the semantics and they have ground truth labels for source data, they compute cross entropy loss given the formula on <img width="45" alt="Ekran Resmi 2021-07-01 12 22 48" src="https://user-images.githubusercontent.com/56236171/124134215-e6efa200-da82-11eb-9c35-9300c5fdfd25.png"> for training on source data.  

<div align="center">
  <img width="300" alt="Ekran Resmi 2021-06-01 18 51 38" src="https://user-images.githubusercontent.com/56236171/124134826-83b23f80-da83-11eb-9af3-852c00eb7ac6.png">
</div> 

After domain alignment, they think the situation as a semi supervised learning problem, since they have labels for some part of the data, <img width="45" alt="Ekran Resmi 2021-07-01 15 53 26" src="https://user-images.githubusercontent.com/56236171/124135896-86f9fb00-da84-11eb-95ed-24c4e12bf773.png">, and there is no ground truth for the other part <img width="25" alt="Ekran Resmi 2021-07-01 15 54 54" src="https://user-images.githubusercontent.com/56236171/124136075-b7da3000-da84-11eb-82dd-ff0bdac2e36f.png">. For the part without labels, target data, they do entropy minimization which makes the predictions on unlabeled data more confident. The loss is computed with the formula:
<div align="center">
<img width="330" alt="Ekran Resmi 2021-06-01 20 41 52" src="https://user-images.githubusercontent.com/56236171/124137745-456a4f80-da86-11eb-9e9e-a990be895ab0.png">
</div>


In this equation <img width="150" alt="Ekran Resmi 2021-07-01 16 15 11" src="https://user-images.githubusercontent.com/56236171/124139099-93338780-da87-11eb-9d33-0db98a65f46c.png"> represents Charbonnier penalty function which penalizes high entropy predictions more than low entropy predictions for <img width="13" alt="Ekran Resmi 2021-07-01 16 22 35" src="https://user-images.githubusercontent.com/56236171/124140248-9aa76080-da88-11eb-88d2-4733f83a6931.png"> > 0.5. For the setting in the paper, this parameter is chosen as 2, which corresponds to the red curve in the figure.
<div align="center">
  <img width="251" alt="Ekran Resmi 2021-06-01 20 45 55" src="https://user-images.githubusercontent.com/56236171/124139989-6764d180-da88-11eb-87ca-d59ea52aecef.png">
</div>
<div align="center">
  Curves of Charbonnier function for different <img width="10" alt="Ekran Resmi 2021-07-01 16 22 35" src="https://user-images.githubusercontent.com/56236171/124140248-9aa76080-da88-11eb-88d2-4733f83a6931.png"> values
</div> 

After scaling this loss function and summing it with cross entropy loss, they train the network <img width="20" alt="Ekran Resmi 2021-07-01 16 30 48" src="https://user-images.githubusercontent.com/56236171/124141559-cbd46080-da89-11eb-8b69-093971dbef2a.png"> from scratch for semantic segmentation using the following formula:
<div align="center">
  <img width="400" alt="Ekran Resmi 2021-06-01 20 40 51" src="https://user-images.githubusercontent.com/56236171/124141066-5bc5da80-da89-11eb-92e7-a2caf343a532.png">
</div>

### Self Supervised Training (SST)
To boost the performance of the method, they also do self-supervised learning and they need pseudo labels for this task. Using predictions of a model as pseudo labels in the next training with SST is self-referential and results in insignificant contribution.  
As a solution, they obtain pseudo labels from the models trained with different <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> values. They pass target images through these models and take the mean prediction as in the equation and behave these labels as ground truth. 
<div align="center">
  <img width="240" alt="Ekran Resmi 2021-06-11 13 44 14" src="https://user-images.githubusercontent.com/56236171/124173392-4497e480-daab-11eb-809b-af2af38e230c.png">
</div>
They call this averaging over predictions of different models process as Multi-band Transfer (MBT). After getting pseudo labels, they add cross entropy loss on target images to the overall loss function and train the network.  
<div align="center">
 <img width="400" alt="Ekran Resmi 2021-06-11 13 45 21" src="https://user-images.githubusercontent.com/56236171/124174049-28487780-daac-11eb-83d9-f5dfc88c6a80.png">
</div>
The whole training process of segmentation network consists of initial training of M models from scratch, and two more rounds of self-supervised training.

### Main Contributions
- Simplicity of domain alignment: Other state-of-the-art methods train a network for image translation. In the proposed method, it is enough to pick a proper <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> value for domain alignment.
- Training a single network: After domain alignment, proposed method only requires training a network, as in a simple semantic segmentation task.
- Computationally less demanding: In FDA paper, there is no computationally demanding implementation such as discriminators or adversarial training. 

## Experimental Setup and Results
### Experiments

In the paper, the authors work on two different domain adaptation scenarios.
- GTA5 to CityScapes: These two datasets have 19 classes in common.
- SYNTHIA to CityScapes: For this scenario, the method is evaluated over 13 and 16 classes.

To show robustness of the method, they trained two segmentation networks <img width="20" alt="Ekran Resmi 2021-07-01 16 30 48" src="https://user-images.githubusercontent.com/56236171/124141559-cbd46080-da89-11eb-8b69-093971dbef2a.png"> separately. The first one is DeepLabv2 with ResNet101 backbone and other one is FCN-8s with VGG16  backbone. Both of these networks are initialized with pretrained weights on ImageNet dataset.

### Results
#### FDA with Single Scale
The first experiment is FDA with single scale on task GTA5 to CityScapes. They train 3 DeepLabV2 networks with the following parameters:  
- <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> = 0.01, 0.05, 0.09
- <img width="30" alt="Ekran Resmi 2021-07-01 21 42 47" src="https://user-images.githubusercontent.com/56236171/124181061-567e8500-dab5-11eb-8c0a-cd4abb88ef78.png"> = 0.005
- <img width="13" alt="Ekran Resmi 2021-07-01 16 22 35" src="https://user-images.githubusercontent.com/56236171/124140248-9aa76080-da88-11eb-88d2-4733f83a6931.png"> = 2.0
<div align="center">
  <img width="856" alt="Ekran Resmi 2021-07-01 21 51 31" src="https://user-images.githubusercontent.com/56236171/124181967-88441b80-dab6-11eb-8644-b907ae45e20a.png">
</div>
<div align="center">
 Results of training from scratch
</div>

We can see the intersection over union (IoU) scores for 19 classes in the table. T=0 represents that the model is trained from scratch. For different <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> values, performance of the network is very similar and that shows robustness of the method with respect to this parameter. In addition, we can see that the best performances (underlined numbers) are almost equally distributed among these three networks.  
These results also show that without entropy minimization, <img width="30" alt="Ekran Resmi 2021-07-01 21 42 47" src="https://user-images.githubusercontent.com/56236171/124181061-567e8500-dab5-11eb-8c0a-cd4abb88ef78.png"> = 0 and <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> = 0.09, FDA outperforms one of the state-of-the-art methods, CycADA. This shows strength of FDA method compared to two-stage image translation based adversarial domain adaptation.
#### Multi-band Transfer (MBT)
The next experiment is for Multi-band Transfer. When they use the predictions of the network with <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> is 0.09, the improvement is only 0.9% compared to the training from scratch scenario as given in the table.
<div align="center">
 <img width="850" alt="Ekran Resmi 2021-07-02 01 21 36" src="https://user-images.githubusercontent.com/56236171/124199736-e501ff00-dad3-11eb-9bf4-3579a6ddc478.png">
</div>
<div align="center">
 Results for SST and MBT
</div>
Without SST, performing only MBT makes a bigger contribution than this. As in the table, the improvement becomes 3.9%. This observation lead the authors to perform MBT to get pseudo labels for SST.

#### Self-supervised Training (SST) with MBT
The last experiment includes SST. Overall training process starts with training of 3 networks with different <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> values from scratch (single scale FDA). Then, averaging over the predictions of these networks (MBT) gives the pseudo labels for SST. Using these labels, SST is performed one round. The second section of the table gives the results for the three networks after performing SST.  
In the end of second round, they perform MBT again to get pseudo labels for the next round. In the last round, these labels are used in the networks and the last result is obtained with MBT. Following table includes the results for overall training process.
<div align="center">
<img width="850" alt="Ekran Resmi 2021-07-02 01 36 39" src="https://user-images.githubusercontent.com/56236171/124201042-3cee3500-dad7-11eb-9bdc-59408c50db06.png">

</div>
<div align="center">
Results of overall process
</div>

One observation from this table is that MBT is the best performer in all the rounds. In addition, performing SST with pseudo labels given by MBT improves performance in each round. Another observation is that network with <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png">=0.09 is the best performer in the first round while it is the worst in the last and the one with <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png">=0.01 becomes the best.  

<div align="center">
<img width="400" alt="Ekran Resmi 2021-06-12 17 15 50" src="https://user-images.githubusercontent.com/56236171/124202880-b4be5e80-dadb-11eb-8283-8d1ff5bdd862.png">
</div>



This situation is illustrated in paper with the given figure. When <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> is small, adapted source data has less chance to cover target dataset than the one with larger <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png">. When pseudo labels are used, adapted source center is closer to target center and variance is smaller. Hence, they conclude that for single scale FDA, it is better to use a larger <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png"> and for MBT, it is better to gradually increase the weight of the network with smaller <img width="14" alt="Ekran Resmi 2021-07-01 20 19 24" src="https://user-images.githubusercontent.com/56236171/124172103-b4a56b00-daa9-11eb-9acf-fd5ab62f443d.png">.

#### Quantitative Results
Quantitative evaluation of the method on GTA5 to CityScapes scenario is given in the table. In the first section of the table, results of ResNet101 backbone is shown and we can see that entropy minimization activated FDA achieves a similar performance with ABStruct and AdvEnt. FDA with SST using MBT outperforms BDL by 4.0 %. Similarly, with VGG16 backbone, FDA outperforms all the other methods.

<div align="center">
<img width="800" alt="Ekran Resmi 2021-06-12 17 20 06" src="https://user-images.githubusercontent.com/56236171/124203428-fe5b7900-dadc-11eb-9166-7b7c1bdf1c5a.png">
</div>
<div align="center">
GTA5 -> CityScapes benchmark
</div>

Following table shows quantitative evaluation on the second scenario, SYNTHIA to CityScapes. This time, FDA is evaluated for 13 classes for ResNet101 and 16 classes for VGG16. For this scenario FDA outperforms the other methods as in GTA5 to CityScapes.
<div align="center">
<img width="770" alt="Ekran Resmi 2021-07-02 02 50 41" src="https://user-images.githubusercontent.com/56236171/124204910-58117280-dae0-11eb-9bd9-cb403173dab7.png">

</div>
<div align="center">
SYNTHIA -> CityScapes benchmark
</div>

#### Qualitative Results
In the paper the authors compare their results with BDL which is the second best performer and uses the same backbone for segmentation. In the first column, there are images from CityScapes dataset and their corresponding ground truth labels. Next to them, we have predictions of BDL and FDA. We can see that predictions of FDA are less noisy and also maintain the structures like poles as can be seen in the last image. In addition, The method performs well on minority classes such as truck and bicycle. 
<div align="center">
<img width="750" alt="Ekran Resmi 2021-07-02 02 59 30" src="https://user-images.githubusercontent.com/56236171/124205457-8cd1f980-dae1-11eb-80e7-d7c946d89b84.png">
</div>
<div align="center">
Comparison of predictions
</div>

## Conclusion
FDA paper proposes a new method for domain alignment which does not require any training. This method showed impressive results just by training a semantic segmentation network with the translated images such that it outperformed CycADA, one of the state-of-the-art methods. Another advantage of the proposed spectral transfer block is that it can be employed by upcoming works in unsupervised domain adaptation field. 
In addition to that, they used SST effectively by employing MBT. Otherwise, SST would be self-referential and improvement would not be significant. With their simple method, they outperform all the other state-of-the-art methods. This study shows that some misalignments due to low-level statistics can be captured by Fourier transform.  
Presentation slides can be found here:[ZeynepGerem_FDA.pdf](https://github.com/zgerem/Seminar/files/6754792/ZeynepGerem_FDA.pdf)


## References
[1] Gabriel J Brostow, Julien Fauqueur, and Roberto Cipolla. Semantic object classes in video: A high-definition ground truth database. Pattern Recognition Letters, 2009.  
[2] Yuhua Chen, Wen Li, and Luc Van Gool. Road: Reality oriented adaptation for semantic segmentation of urban scenes. In ECCV, 2018.  
[3] Stephan R Richter, Vibhav Vineet, Stefan Roth, and Vladlen Koltun. Playing for data: Ground truth from computer games. In ECCV, 2016.  
[4] German Ros, Laura Sellart, Joanna Materzynska, David Vazquez, and Antonio M Lopez. The synthia dataset: A large collection of synthetic images for semantic segmentation of urban scenes. In CVPR, 2016.  
[5]  Judy Hoffman, Eric Tzeng, Taesung Park, Jun-Yan Zhu, Phillip Isola, Kate Saenko, Alexei Efros, and Trevor Darrell. Cycada: Cycle-consistent adversarial domain adaptation. In ICML, 2018.  
[6] Yunsheng Li, Lu Yuan, and Nuno Vasconcelos. Bidirectional learning for domain adaptation of semantic segmentation. In CVPR, 2019.
