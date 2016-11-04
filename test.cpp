//#define _DEBUG_INIT_
//#define _DEBUG_MINST_
//#define _DEBUG_IN_MP_
//#define _DEBUG_Y_
//#define _DEBUG_T_

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include "CNN.h"
#include "MINST/MINST.h"

float right=0;

Convolutional_Neural_Network CNN;

void input_minst(MinstImg input){
	for (int i=0;i<28;i++){
		for (int j=0;j<28;j++){
			CNN.INPUT.y.d[0][i][j]=input.ImgData[i][j];
		}
	}
}

int sort(float *data){
	int id=0;
	float max=data[0];
	for (int i=0;i<10;i++){
		if (data[i]>max){
			max=data[i];
			id=i;
		}
	}
	return id;
}

int main(){
	int timestart;
	printf("===========\nMINST_READ\n===========\n");
	printf("--\nReading training images.\n");
	ImgArr train_img = read_Img("./MINST/train-images.idx3-ubyte");
	printf("Read training images finished.\n");
	printf("Training images data:\n");
	printf("%d images.\n",train_img->ImgNum);
	printf("--\nReading testing images.\n");
	printf("Size : %d*%d.\n",train_img->ImgPtr[0].c,train_img->ImgPtr[0].r);
	ImgArr test_img = read_Img("./MINST/t10k-images.idx3-ubyte");
	printf("Read testing images finished.\n");
	printf("Testing images data:\n");
	printf("%d images.\n",test_img->ImgNum);
	printf("Size : %d*%d.\n",test_img->ImgPtr[0].c,test_img->ImgPtr[0].r);

	printf("--\nReading training lables.\n");
	LabelArr train_label = read_Lable("./MINST/train-labels.idx1-ubyte");
	printf("Read training lables finished.\n");
	printf("Training lables data:\n");
	printf("%d lables.\n",train_label->LabelNum);
	printf("--\nReading testing lables.\n");
	LabelArr test_label = read_Lable("./MINST/t10k-labels.idx1-ubyte");
	printf("Read testing lables finished.\n");
	printf("Testing lables data:\n");
	printf("%d lables.\n",test_label->LabelNum);

	printf("===========\nCNN_INIT\n===========\n");

	srand(time(NULL));

	CNN.init();

#ifdef	_DEBUG_MINST_
	printf("===========\nMINST_SHOW\n===========\n");
	printf("train_img\n");
	for (int i=0;i<28;i++){
		for (int j=0;j<28;j++){
			std::cout<<train_img->ImgPtr[0].ImgData[i][j]<<'|';
		}
		std::cout<<std::endl;
	}
	printf("train_label\n");
	for (int i=0;i<10;i++){
		std::cout<<train_label->LabelPtr[0].LabelData[i]<<'|';
	}
#endif

	std::cout<<std::endl;
	for(int j=0;j<1000;j++){
	int time_s=time(NULL);
	for (int i=0;i<60000;i++){
		float error=0;
		input_minst(train_img->ImgPtr[i]);
		CNN.train(train_label->LabelPtr[i].LabelData,0.5,0.5,0.5);
		if ((i%5000)==0){
			std::cerr<<i<<'|'<<sort(train_label->LabelPtr[i].LabelData)<<'|'<<(time(NULL)-time_s)<<std::endl;
			for(int l=0;l<500;l++){
//			float erro=0;
			input_minst(test_img->ImgPtr[l]);
			CNN.calculate();
//			for(int j=0;j<10;j++)
//			{
//				erro+=(test_label->LabelPtr[0].LabelData[j]-CNN.FC_9.y[j])*(test_label->LabelPtr[0].LabelData[j]-CNN.FC_9.y[j]);
//			}
			if(sort(test_label->LabelPtr[l].LabelData)==sort(CNN.FC_9.y)){
				right=right+1;
			}
//			std::cout<<erro<<'|'<<sort(test_label->LabelPtr[0].LabelData)<<'|'<<sort(CNN.FC_9.y)<<'|'<<sort(train_label->LabelPtr[i].LabelData)<<std::endl;
			}
			std::cout<<right/500<<std::endl;
			right=0;
			time_s=time(NULL);
		}
	}

//	for (int i=0;i<3;i++){
//		input_minst(test_img->ImgPtr[i]);
//		CNN.calculate();
//		for(int j=0;j<CNN.C_1.m;j++){
//			for(int k=0;k<CNN.C_1.n;k++){
//				std::cout<<CNN.C_1.y.d[0][j][k]<<'|';
//			}
//			std::cout<<std::endl;
//		}
//		std::cout<<"==================="<<std::endl;
//		for(int j=0;j<CNN.MP_2.m;j++)
		{
//			for(int k=0;k<CNN.MP_2.n;k++){
//				printf("%f,",CNN.MP_2.y.d[0][j][k]);
//			}

		}
//		std::cout<<"==================="<<std::endl;
//		for(int j=0;j<10;j++)
		{
//			printf("%f,",CNN.FC_9.y[j]);
		}
//		std::cout<<"==================="<<std::endl;
//		std::cout<<'\n';
//		if (sort(test_label->LabelPtr[i].LabelData)==sort(CNN.FC_8.y)){
//			printf("good\n");
//			right+=1;
//		}
//	}
//	printf("%f\n",right);
	}
	std::cout<<std::endl<<"OK!"<<std::endl;

}
