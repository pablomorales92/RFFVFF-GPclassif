Code for the methods RFF-GPC and VFF-GPC. They use Fourier features approximations and variational inference to scale up GP classfication. The former samples the Fourier frequencies (in order to approximate the RBF kernel), whereas the latter optimizes over them. Both methods were introduced and tested in the context of remote sensing image classification.

Full reference:\
Morales-Álvarez P., Pérez-Suay A., Molina R., Camps-Valls G.\
Remote Sensing Image Classification With Large-Scale Gaussian Processes\
IEEE Transactions on Geoscience and Remote Sensing, 2018
DOI: 10.1109/TGRS.2017.2758922

## Abstract
Current remote sensing image classification problems have to deal with an unprecedented amount of heterogeneous and complex data sources. Upcoming missions will soon provide large data streams that will make land cover/use classification difficult. Machine-learning classifiers can help at this, and many methods are currently available. A popular kernel classifier is the Gaussian process classifier (GPC), since it approaches the classification problem with a solid probabilistic treatment, thus yielding confidence intervals for the predictions as well as very competitive results to the state-of-the-art neural networks and support vector machines. However, its computational cost is prohibitive for large-scale applications, and constitutes the main obstacle precluding wide adoption. This paper tackles this problem by introducing two novel efficient methodologies for GP classification. We first include the standard random Fourier features approximation into GPC, which largely decreases its computational cost and permits large-scale remote sensing image classification. In addition, we propose a model which avoids randomly sampling a number of Fourier frequencies and alternatively learns the optimal ones within a variational Bayes approach. The performance of the proposed methods is illustrated in complex problems of cloud detection from multispectral imagery and infrared sounding data. Excellent empirical results support the proposal in both computational cost and accuracy.

## Citation
@article{morales2018remote,\
  title={Remote sensing image classification with large-scale gaussian processes},\
  author={Morales-{\\'A}lvarez, Pablo and P{\\'e}rez-Suay, Adri{\\'a}n and Molina, Rafael and Camps-Valls, Gustau},\
  journal={IEEE Transactions on Geoscience and Remote Sensing},\
  volume={56},\
  number={2},\
  pages={1103--1114},\
  year={2018},\
  publisher={IEEE}\
}
