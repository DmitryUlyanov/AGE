#!/bin/bash
cd pretrained
array=( celeba_e celeba_g cifar10_e cifar10_g imagenet_e imagenet_g )
for what in "${array[@]}"
do
    wget -nc -O ${what}.pth https://github.com/DmitryUlyanov/dmitryulyanov.github.io/blob/master/assets/age/pretrained/${what}.pth?raw=true 
done