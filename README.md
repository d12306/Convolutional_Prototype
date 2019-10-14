# Convolutional_Prototype

main training entry: mnist_gcpl.py. We can change the model architecture, dataset, loss type and so on.   
Note that if we use the dot product as the comparison metric, the prototype loss should be set to 0. Otherwise, the model can not converge.

model structure: nets.py. Network for MNIST is the same as this [CVPR2018 paper](https://github.com/YangHM/Convolutional-Prototype-Learning), Momentum optimizer, initial lr 0.1, decay step 60 (by 0.5) to train the distance based multi prototype loss. Adam 3e-4 to train the softmax loss. 

Network for Cifar10 is ResNet-20 (300 epoches) with data augmentation, Momentum optimizer, initial lr 0.1, decay step 60 (by 0.5) to train the softmax loss and the distance based multi prototype loss (also with weight decay 0.0002).

## Experiment Results

| Dataset                              | Method  | Accuracy |
| -------------------------------------- | ------------- | -------- | 
| MNIST     | Softmax      |  98.65% | 
|MNIST| Multi-Proto (1)distance    |  99.58% |
|MNIST| Multi-Proto (5)distance    |  99.61% |
|MNIST| Multi-Proto (10)distance    |  99.65% |
|MNIST| Multi-Proto (1)dot-product    |   |
|MNIST| Multi-Proto (5)dot-product    |   |
|MNIST| Multi-Proto (10)dot-product     |   |
| Cifar10    | Softmax      | 90.86%  |
|Cifar10| Multi-Proto (5)distance   | 90.96%  |
|Cifar10| Multi-Proto (1)distance   | 90.73%  |
|Cifar10| Multi-Proto (10)distance   |  90.96% |
|Cifar10| Multi-Proto (5)dot-product    |  |
|Cifar10| Multi-Proto (1)dot-product  |  |
|Cifar10| Multi-Proto (10)dot-product   | |
| Cifar100     | Softmax       |   | 
|Cifar100| Multi-Proto (5)-distance    |  |
|Cifar100| Multi-Proto (1)-distance    |   |
|Cifar100| Multi-Proto (10)-distance    |   |
|ImageNet   | Softmax       |    | 
|ImageNet | Multi-Proto    |   |

