# Convolutional_Prototype

main training entry: mnist_gcpl.py. We can change the model architecture, dataset, loss type and so on.   
Note that if we use the dot product as the comparison metric, the prototype loss should be set to 0. Otherwise the model can not converge.

model structure: nets.py. Network for MNIST is the same as this [CVPR2018 paper](https://github.com/YangHM/Convolutional-Prototype-Learning), Network for Cifar10 is ResNet-20 with data augmentation, Momentum optimizer, initial lr 0.1, decay step 60 (by 0.5).

## Experiment Results

| Dataset                              | Method  | Accuracy |
| -------------------------------------- | ------------- | -------- | 
| MNIST     | Softmax      |   | 
|MNIST| Multi-Proto (1)   |   |
|MNIST| Multi-Proto (5)   |   |
|MNIST| Multi-Proto (10)   |   |
| Cifar10    | Softmax      | 90.86%  |
|Cifar10| Multi-Proto (5)   | 90.96%  |
|Cifar10| Multi-Proto (1)   |   |
|Cifar10| Multi-Proto (10)   |   |
| Cifar100     | Softmax        |   | 
|Cifar100| Multi-Proto (5)   |  |
|Cifar100| Multi-Proto (1)   |   |
|Cifar100| Multi-Proto (10)   |   |
|ImageNet   | Softmax       |    | 
|ImageNet | Multi-Proto    |   |

