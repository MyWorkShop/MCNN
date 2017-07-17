// #define _DEBUG_INIT_
//#define _DEBUG_MNIST_
// #define _DEBUG_Y_
// #define _DEBUG_T_

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include "MNIST/MNIST.h"
#include "CNN/random.h"
#include "CNN/data.h"
#include "CNN/CNN.h"
#include "CNN/MCNN.h"
#include "CNN/SCNN.h"
#include "CNN/CSCLayer.h"
#include "CNN/CSCCNN.h"

float right = 0;

FILE *output;

// Middle_Convolutional_Neural_Network *CNN;
CSC_Convolutional_Neural_Network *CNN;

void input_MNIST(MNISTImg input, int index)
{
    for (int i = 0; i < 28; i++)
    {
	for (int j = 0; j < 28; j++)
	{
	    // CNN[index].INPUT.y.d[0][i][j] = input.ImgData[i][j];
	    CNN[index].INPUT.y.d[0][i + 2][j + 2] = input.ImgData[i][j];
	}
    }
}

int sort(float *data)
{
    int id = 0;
    float max = data[0];
    for (int i = 0; i < 10; i++)
    {
	if (data[i] > max)
	{
	    max = data[i];
	    id = i;
	}
    }
    return id;
}

int main()
{
    int time_s;
    int t_num = 4;
    int id = 0;
    float eta = 0.001;
    float eta_min = 0.00003;
    float eta_m = 0.993;
    omp_set_num_threads(t_num);
    std::cout << "The threads will be use:" << t_num << '\n';
    output = fopen("output.csv", "a");
    ImgArr train_img = read_Img("./MNIST/train-images.idx3-ubyte");
    ImgArr test_img = read_Img("./MNIST/t10k-images.idx3-ubyte");
    LabelArr train_label = read_Lable("./MNIST/train-labels.idx1-ubyte");
    LabelArr test_label = read_Lable("./MNIST/t10k-labels.idx1-ubyte");

    srand(time(NULL));

    // CNN = new Middle_Convolutional_Neural_Network[t_num];
    CNN = new CSC_Convolutional_Neural_Network[t_num];
    for (int i = 0; i < t_num; i++)
    {
	CNN[i].init();
    }

#ifdef _DEBUG_MNIST_
    printf("train_img\n");
    for (int i = 0; i < 28; i++)
    {
	for (int j = 0; j < 28; j++)
	{
	    if (train_img->ImgPtr[0].ImgData[i][j] > 0.3)
	    {
		std::cout << "*";
	    }
	    else
	    {
		std::cout << " ";
	    }
	}
	std::cout << std::endl;
    }
    printf("train_label\n");
    for (int i = 0; i < 10; i++)
    {
	if (train_label->LabelPtr[0].LabelData[i] > 0.3)
	{
	    std::cout << "*|";
	}
	else
	{
	    std::cout << " |";
	}
    }
#endif

    std::cout << std::endl;
    time_s = time(NULL);
    for (int j = 0; j < 500000; j++)
    {
#pragma omp parallel private(id)
	{
	    id = omp_get_thread_num();
	    if ((j % 500) == 0)
	    {
#pragma omp for
		for (int l = 0; l < 10000; l++)
		{
		    input_MNIST(test_img->ImgPtr[l], id);
		    CNN[id].calculate();
		    if (sort(test_label->LabelPtr[l].LabelData) == sort(CNN[id].FC_9.y))
		    {
			right = right + 1;
		    }
		}
#pragma omp barrier
#ifdef _DEBUG_MNIST_
#pragma omp master
		{
		    for (int i = 0; i < 10; i++)
		    {
			if (CNN[id].FC_9.y[i] > 0.1)
			{
			    std::cout << CNN[id].FC_9.y[i] << "|";
			}
			else
			{
			    std::cout << " |";
			}
		    }
		    std::cout << std::endl;
		}
#endif
#pragma omp master
		{
		    if (eta > eta_min)
		    {
			eta = eta * eta_m;
		    }
		    std::cerr << j / 500 << '|' << (time(NULL) - time_s) << '|' << eta << std::endl;
		    std::cout << right / 10000 << std::endl;
		    fprintf(output, "%d,%f\n", j / 500, right / 10000);
		    right = 0;
		    time_s = time(NULL);
		}
	    }
#pragma omp barrier
#pragma omp for
	    for (int i = 0; i < 120; i++)
	    {
		input_MNIST(train_img->ImgPtr[i + ((j % 500) * 120)], id);
		CNN[id].train(train_label->LabelPtr[i + ((j % 500) * 120)].LabelData);
	    }
#pragma omp barrier
#pragma omp master
	    {
		for (int i = 0; i < t_num; i++)
		{
		    CNN[0].change_weight(&(CNN[i]), eta);
		}
	    }
#pragma omp barrier
	    if (id == 0)
	    {
		goto end_copy;
	    }
	    CNN[id].copy_weight(&(CNN[0]));
	end_copy:;
	}
    }
    std::cout << std::endl
	      << "OK!" << std::endl;
}
