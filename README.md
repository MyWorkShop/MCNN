# Convolutional neural networks
### By Huang Tao

***
##OpenMP supported
Use ./topm.cpp

##Layers
* Convolutional layer
* Max pooling layer
* Fully connected layer

***

##Test 
    MINST Database
    90% correct rate 

![rate](https://bbs.kechuang.org/r/271583)

***

##Usages
```c
ImgArr train_img = read_Img("./MINST/train-images.idx3-ubyte");
ImgArr test_img = read_Img("./MINST/t10k-images.idx3-ubyte");
```

Read MINST image data.

```c
LabelArr train_label = read_Lable("./MINST/train-labels.idx1-ubyte");
LabelArr test_label = read_Lable("./MINST/t10k-labels.idx1-ubyte");
```

Read MINST label data.

```c
input_minst(test_img->ImgPtr[l]);
```

Input the data to the neural network from MINST database.

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
    
Train the neural network with the label from MINST database.(input the data first)

***

##Change log

