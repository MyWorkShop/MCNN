# Convolutional neural networks
### By Huang Tao

## OpenMP has been supported

## Dropout has been supported

## Layers
* Convolutional layer(support dropout)
* Max pooling layer
* Fully connected layer(support dropout)
* Softmax layer


## Test 
    MNIST Database
    99.11% correct rate 

## Usages
```c
ImgArr train_img = read_Img("./MNIST/train-images.idx3-ubyte");
ImgArr test_img = read_Img("./MNIST/t10k-images.idx3-ubyte");
```

Read MNIST image data.

```c
LabelArr train_label = read_Lable("./MNIST/train-labels.idx1-ubyte");
LabelArr test_label = read_Lable("./MNIST/t10k-labels.idx1-ubyte");
```

Read MNIST label data.

```c
input_MNIST(test_img->ImgPtr[l]);
```

Input the data to the neural network from MNIST database.

```c
CNN.calculate();
```

Calculate the neural network.()input the data first)

```c
sort(test_label->LabelPtr[l].LabelData)
```

Get the output index.

```c
CNN.train(train_label->LabelPtr[i].LabelData,0.0005,0.0005,0.0005);
```
    
Train the neural network with the label from MNIST database.(input the data first)

***

##Change log

