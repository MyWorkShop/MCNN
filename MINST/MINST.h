typedef struct MinstImg{
	int c;
	int r;
	float** ImgData;
}MinstImg;

typedef struct MinstImgArr{
	int ImgNum;
	MinstImg* ImgPtr;
}*ImgArr;

typedef struct MinstLabel{
	int l;
	float* LabelData;
}MinstLabel;

typedef struct MinstLabelArr{
	int LabelNum;
	MinstLabel* LabelPtr;
}*LabelArr;


int ReverseInt(int i)   
{  
	unsigned char ch1, ch2, ch3, ch4;  
	ch1 = i & 255;  
	ch2 = (i >> 8) & 255;  
	ch3 = (i >> 16) & 255;  
	ch4 = (i >> 24) & 255;  
	return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;  
}


ImgArr read_Img(const char* filename) // 读入图像
{
    FILE  *fp=NULL;
    fp=fopen(filename,"rb");
    if(fp==NULL)
        printf("open file failed\n");
    assert(fp);

    int magic_number = 0;  
    int number_of_images = 0;  
    int n_rows = 0;  
    int n_cols = 0;  
    //从文件中读取sizeof(magic_number) 个字符到 &magic_number  
    fread((char*)&magic_number,sizeof(magic_number),1,fp); 
    magic_number = ReverseInt(magic_number);  
    //获取训练或测试image的个数number_of_images 
    fread((char*)&number_of_images,sizeof(number_of_images),1,fp);  
    number_of_images = ReverseInt(number_of_images);    
    //获取训练或测试图像的高度Heigh  
    fread((char*)&n_rows,sizeof(n_rows),1,fp); 
    n_rows = ReverseInt(n_rows);                  
    //获取训练或测试图像的宽度Width  
    fread((char*)&n_cols,sizeof(n_cols),1,fp); 
    n_cols = ReverseInt(n_cols);  
    //获取第i幅图像，保存到vec中 
    int i,r,c;

    // 图像数组的初始化
    ImgArr imgarr=(ImgArr)malloc(sizeof(MinstImgArr));
    imgarr->ImgNum=number_of_images;
    imgarr->ImgPtr=(MinstImg*)malloc(number_of_images*sizeof(MinstImg));

    for(i = 0; i < number_of_images; ++i)  
    {  
        imgarr->ImgPtr[i].r=n_rows;
        imgarr->ImgPtr[i].c=n_cols;
        imgarr->ImgPtr[i].ImgData=(float**)malloc(n_rows*sizeof(float*));
        for(r = 0; r < n_rows; ++r)      
        {
            imgarr->ImgPtr[i].ImgData[r]=(float*)malloc(n_cols*sizeof(float));
            for(c = 0; c < n_cols; ++c)
            { 
                // 因为神经网络用float型计算更为精确，这里我们将图像像素转为浮点型
                unsigned char temp = 0;  
                fread((char*) &temp, sizeof(temp),1,fp); 
                imgarr->ImgPtr[i].ImgData[r][c]=(float)temp/255.0;
            }  
        }    
    }

    fclose(fp);
    return imgarr;
}


LabelArr read_Lable(const char* filename)// 读入图像
{
    FILE  *fp=NULL;
    fp=fopen(filename,"rb");
    if(fp==NULL)
        printf("open file failed\n");
    assert(fp);

    int magic_number = 0;  
    int number_of_labels = 0; 
    int label_long = 10;

    //从文件中读取sizeof(magic_number) 个字符到 &magic_number  
    fread((char*)&magic_number,sizeof(magic_number),1,fp); 
    magic_number = ReverseInt(magic_number);  
    //获取训练或测试image的个数number_of_images 
    fread((char*)&number_of_labels,sizeof(number_of_labels),1,fp);  
    number_of_labels = ReverseInt(number_of_labels);    

    int i,l;

    // 图像标记数组的初始化
    LabelArr labarr=(LabelArr)malloc(sizeof(MinstLabelArr));
    labarr->LabelNum=number_of_labels;
    labarr->LabelPtr=(MinstLabel*)malloc(number_of_labels*sizeof(MinstLabel));

    for(i = 0; i < number_of_labels; ++i)  
    {
        // 数据库内的图像标记是一位，这里将图像标记变成10位，10位中只有唯一一位为1，为1位即是图像标记  
        labarr->LabelPtr[i].l=10;
        labarr->LabelPtr[i].LabelData=(float*)calloc(label_long,sizeof(float));
        unsigned char temp = 0;  
        fread((char*) &temp, sizeof(temp),1,fp); 
        labarr->LabelPtr[i].LabelData[(int)temp]=1.0;    
    }

    fclose(fp);
    return labarr;    
}
