//#define _DEBUG_INIT_
//#define _DEBUG_MNIST_
//#define _DEBUG_IN_MP_
//#define _DEBUG_Y_
//#define _DEBUG_T_

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include "CNN.h"
#include "MNIST/MNIST.h"
#include <omp.h>

float right = 0;

FILE *output;

Convolutional_Neural_Network *CNN;

void input_MNIST(MNISTImg input, int index)
{
	for (int i = 0; i < 28; i++)
	{
#pragma simd
		for (int j = 0; j < 28; j++)
		{
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
	int t_num = 8;
	int id = 0;
	float eta = 0.001;
	float eta_min = 0.00003;
	float eta_m = 0.993;
	omp_set_num_threads(t_num);
	output = fopen("output.csv", "a");
	std::cout << "The threads will be use:" << t_num << '\n';
	printf("===========\nMNIST_READ\n===========\n");
	printf("--\nReading training images.\n");
	ImgArr train_img = read_Img("./MNIST/train-images.idx3-ubyte");
	printf("Read training images finished.\n");
	printf("Training images data:\n");
	printf("%d images.\n", train_img->ImgNum);
	printf("--\nReading testing images.\n");
	printf("Size : %d*%d.\n", train_img->ImgPtr[0].c, train_img->ImgPtr[0].r);
	ImgArr test_img = read_Img("./MNIST/t10k-images.idx3-ubyte");
	printf("Read testing images finished.\n");
	printf("Testing images data:\n");
	printf("%d images.\n", test_img->ImgNum);
	printf("Size : %d*%d.\n", test_img->ImgPtr[0].c, test_img->ImgPtr[0].r);

	printf("--\nReading training lables.\n");
	LabelArr train_label = read_Lable("./MNIST/train-labels.idx1-ubyte");
	printf("Read training lables finished.\n");
	printf("Training lables data:\n");
	printf("%d lables.\n", train_label->LabelNum);
	printf("--\nReading testing lables.\n");
	LabelArr test_label = read_Lable("./MNIST/t10k-labels.idx1-ubyte");
	printf("Read testing lables finished.\n");
	printf("Testing lables data:\n");
	printf("%d lables.\n", test_label->LabelNum);

	printf("===========\nCNN_INIT\n===========\n");

	srand(time(NULL));

	CNN = new Convolutional_Neural_Network[t_num];
	for (int i = 0; i < t_num; i++)
	{
		CNN[i].init();
	}
	for (int i = 0; i < 20; i++)
	{
		//		std::cout<<R()<<'\n';
	}

#ifdef _DEBUG_MNIST_
	printf("===========\nMNIST_SHOW\n===========\n");
	printf("train_img\n");
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			std::cout << train_img->ImgPtr[0].ImgData[i][j] << '|';
		}
		std::cout << std::endl;
	}
	printf("train_label\n");
	for (int i = 0; i < 10; i++)
	{
		std::cout << train_label->LabelPtr[0].LabelData[i] << '|';
	}
#endif

	std::cout << std::endl;
	time_s = time(NULL);
	for (int j = 0; j < 500000; j++)
	{
#pragma omp parallel private(id)
		{
			id = omp_get_thread_num();
			//std::cerr<<id<<std::endl;
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
						//					std::cerr<<right<<'|';
					}
				}
#pragma omp barrier
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
//std::cerr<<"barrier"<<std::endl;
#pragma omp for
			for (int i = 0; i < 120; i++)
			{
				input_MNIST(train_img->ImgPtr[i + ((j % 500) * 120)], id);
				CNN[id].train(train_label->LabelPtr[i + ((j % 500) * 120)].LabelData);
				//				std::cerr<<"train"<<std::endl;
			}
#pragma omp barrier
//std::cerr<<"barrier"<<std::endl;
#pragma omp master
			{
				//std::cerr<<"master"<<std::endl;
				for (int i = 0; i < t_num; i++)
				{
					//std::cerr<<CNN[i].C_1.d_w.d[0][0][0][0]<<std::endl;
					CNN[0].change_weight(&(CNN[i]), eta);
				}
			}
#pragma omp barrier
			//std::cerr<<id<<std::endl;
			if (id == 0)
			{
				//std::cerr<<"id==0"<<std::endl;
				goto end_copy;
			}
			//std::cerr<<"copy"<<std::endl;
			CNN[id].copy_weight(&(CNN[0]));
		end_copy:;
		}
	}
	std::cout << std::endl
			  << "OK!" << std::endl;
}
