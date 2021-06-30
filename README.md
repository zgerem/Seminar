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
<img width="120" alt="Ekran Resmi 2021-06-30 23 54 05" src="https://user-images.githubusercontent.com/56236171/124036639-9afb1980-d9fe-11eb-911a-b9a117892a5d.png"> represents source domain dataset


