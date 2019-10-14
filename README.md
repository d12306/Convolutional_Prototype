# Convolutional_Prototype

main training entry: mnist_gcpl.py. We can change the model architecture, dataset, loss type and so on.   
Note that if we use the dot product as the comparison metric, the prototype loss should be set to 0. Otherwise the model can not converge.

model structure: nets.py

## Experiment Results

| Dataset                              | Method  | Accuracy |
| -------------------------------------- | ------------- | -------- | 
| MNIST     | Softmax      |   | 
| MNIST     | Multi-Proto    |   | 
| Cifar10    | Softmax      |   |
|Cifar10| Multi-Proto    |   |
| Cifar100     | Softmax        |   | 
|Cifar100 | Multi-Proto    |   |
|ImageNet   | Softmax       |    | 
|ImageNet | Multi-Proto    |   |

