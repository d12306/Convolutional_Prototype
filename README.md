# Convolutional_Prototype

main training entry: mnist_gcpl.py. We can change the model architecture, dataset, loss type and so on.   
Note that if we use the dot product as the comparison metric, the prototype loss should be set to 0. Otherwise, the model can not converge.

model structure: nets.py. Network for MNIST is the same as this [CVPR2018 paper](https://github.com/YangHM/Convolutional-Prototype-Learning), Momentum optimizer, initial lr 0.1, decay step 60 (by 0.5) to train the distance based multi prototype loss. Adam 3e-4 to train the softmax loss. 

Network for Cifar10 is ResNet-20 (with double filters, 300 epoches) with data augmentation, Momentum optimizer, initial lr 0.1, decay step 60 (by 0.5) to train the softmax loss and the distance based multi prototype loss (also with weight decay 0.0002).

Network for Cifar100 is ResNet-56 (with double filters, 300 epoches) with data augmentation, Momentum optimizer, initial lr 0.1, decay step 150 and 225 (by 0.1) to train the softmax loss and the distance based multi prototype loss. 

## Experiment Results

| Dataset                              | Method  | Accuracy |
| -------------------------------------- | ------------- | -------- | 
| MNIST     | Softmax      |  98.75% | 
|MNIST| Multi-Proto (1)distance    |  99.67% |
|MNIST| Multi-Proto (5)distance    |  99.73% |
|MNIST| Multi-Proto (10)distance    |  99.66% |
|MNIST| Multi-Proto (1)dot-product    |  99.62% |
|MNIST| Multi-Proto (5)dot-product    |  99.69% |
|MNIST| Multi-Proto (10)dot-product     |99.70%   |
| Cifar10    | Softmax      | 93.27%  |
|Cifar10| Multi-Proto (1)distance   | 93.43%  |
|Cifar10| Multi-Proto (5)distance   | 92.41%  |
|Cifar10| Multi-Proto (10)distance   |  91.79% |
|Cifar10| Multi-Proto (1)dot-product  | 93.66% |
|Cifar10| Multi-Proto (5)dot-product    |  91.80%|
|Cifar10| Multi-Proto (10)dot-product   |91.32% |
| Cifar100     | Softmax       |   73.31%| 
|Cifar100| Multi-Proto (1)-distance    | 73.78%  |
|Cifar100| Multi-Proto (5)-distance    | 71.95% |
|Cifar100| Multi-Proto (10)-distance    | 70.96%  |
|Cifar100| Multi-Proto (1)dot-product  | 73.90%|
|Cifar100| Multi-Proto (5)dot-product    | 71.34% |
|Cifar100| Multi-Proto (10)dot-product   |71.31% |
|ImageNet   | Softmax       |    | 
|ImageNet | Multi-Proto    |   |

