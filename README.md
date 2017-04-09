This repository contains code for the paper

["Adversarial Generator-Encoder Networks"](http://sites.skoltech.ru/app/data/uploads/sites/25/2017/04/AGE.pdf) by Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky.

![](data/readme_pics/age.png)

For now only evaluation code and the models from paper are here, training code will be added later this week.

## Pretrained models

1) First install dev version of pyTorch (see manual [here](INSTALL.md)) and make sure you have `jupyter notebook` ready.

2) Then download the models with the script:
```
bash download_pretrained.sh
```

3) Run `jupyter notebook` and go through `evaluate.ipynb`.


Here is an example of samples and reconstructions for `imagenet`, `celeba` and `cifar10` datasets.

#### Celeba

|Samples    |Reconstructions|
|:---------:|:-------------:|
|![](data/readme_pics/celeba_samples.png) | ![](data/readme_pics/celeba_reconstructions.png) |

#### Cifar10

|Samples    |Reconstructions|
|:---------:|:-------------:|
|![](data/readme_pics/cifar10_samples.png) | ![](data/readme_pics/cifar10_reconstructions.png) |

#### Tiny ImageNet

|Samples    |Reconstructions|
|:---------:|:-------------:|
|![](data/readme_pics/imagenet_samples.png) | ![](data/readme_pics/imagenet_reconstructions.png) |
