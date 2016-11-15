#include <stdlib.h>
#include <math.h>

//类预定义

class tube;
class cube;
class mat;
class array;
class Input_Layer;
class Convolutional_Layer;
class Max_Pooling_Layer;
class Fully_Connected_Layer;

//随机函数

float R(){
	if ((rand()%2)==0){
		return -1;
	}else{
		return 1;
	}
}

float r(){
	return rand()/(RAND_MAX+1.0);
}

// 输出函数

namespace sigmoid{
	float y(float x){
		return 1/(1+exp(-x));
	}

	float df(float y){
		return y*(1-y);
	}
}

//数据存储类

//4 aixs
class tube{
public:
	float ****d;

	void init(int x,int y,int z,int a){
		d=new float***[x];
		for (int i=0;i<x;i++){
			d[i]=new float**[y];
			for (int j=0;j<y;j++){
				d[i][j]=new float*[z];
				for(int k=0;k<z;k++){
					d[i][j][k]=new float[a];
				}
			}
		}
	}

	void init(int x,int y,int z,int a,void *r_init){
		d=new float***[x];
		for (int i=0;i<x;i++){
			d[i]=new float**[y];
			for (int j=0;j<y;j++){
				d[i][j]=new float*[z];
				for(int k=0;k<z;k++){
					d[i][j][k]=new float[a];
					for(int l=0;l<a;l++){
						d[i][j][k][l]=R();
					}
				}
			}
		}
	}
};

//3 aixs
class cube{
public:
	float ***d;

	void init(int x,int y,int z){
		d=new float**[x];
		for (int i=0;i<x;i++){
			d[i]=new float*[y];
			for (int j=0;j<y;j++){
				d[i][j]=new float[z];
			}
		}
	}

	void init(int x,int y,int z,void *r_init){
		d=new float**[x];
		for (int i=0;i<x;i++){
			d[i]=new float*[y];
			for (int j=0;j<y;j++){
				d[i][j]=new float[z];
				for(int k=0;k<z;k++){
					d[i][j][k]=R();
				}
			}
		}
	}
};

//2 aixs
class mat{
public:
	float **d;

	void init(int x,int y){
		d=new float*[x];
		for (int i=0;i<x;i++){
			d[i]=new float[y];
		}
	}

	void init(int x,int y,void *r_init){
		d=new float*[x];
		for (int i=0;i<x;i++){
			d[i]=new float[y];
			for (int j=0;j<y;j++){
				d[i][j]=R();
			}
		}
	}
};

class array{
public:
	float *d;

	void init(int num){
		d=new float[num];
	}
	
	void init(int num,void *r_init){
		d=new float[num];
		for (int i=0;i<num;i++){
			d[i]=R();
		}
	}
};

//层类

class Input_Layer{
public:
	cube y;
	int num;
	int m,n;

	void init(int num_pic,int a,int b){
		num=num_pic;
		m=a;
		n=b;
		y.init(num,m,n);
	}
};

class Convolutional_Layer{
public:
	tube w;
	int m,n;
	int a,b;
	cube y;
	cube dleta;
	cube bias;
	int num;
	void *last_layer;
	int last_m,last_n;
	int last_num;
	Max_Pooling_Layer *next_layer;
	int next_num;
	int next_a,next_b;
	bool s;

	float core(int x_aixs,int y_aixs,int num_source,int num_output);

	void init_1(void *last,int num_pics,int a_core,int b_core,bool start);
	void init_2(Max_Pooling_Layer *next);

	void calculate_y(){
		for(int i=0;i<num;i++){
			for(int j=0;j<m;j++){
				for(int k=0;k<n;k++){
					float sum=0;
					for(int l=0;l<last_num;l++){
						sum=sum+core(j,k,l,i);
					}
					y.d[i][j][k]=sigmoid::y(sum+bias.d[i][j][k]);
				}
			}
		}
	}

	void calculate_dleta();

	void change_weight(float eta);
};

class Max_Pooling_Layer{
public:
	cube beta,bias;
	Convolutional_Layer *last_layer;
	int m,n;
	int a,b;
	int num;
	void *next_layer;
	int next_m,next_n;
	int next_a,next_b;
	int next_num;
	bool Fully_Connected;
	cube y;
	cube d;
	cube dleta;

	float max(int no,int x_aixs,int y_aixs){
		float max_float=last_layer->y.d[no][x_aixs*a][y_aixs*b];
		for (int i=0;i<a;i++){
			for (int j=0;j<b;j++){
//				#ifdef	_DEBUG_IN_MP_
//					printf("max_float=%f\n",max_float);
//					printf("last_layer->y=%f\n",last_layer->y.d[no][x_aixs*a+i][y_aixs*b+j]);
//				#endif
				if(last_layer->y.d[no][x_aixs*a+i][y_aixs*b+j]>=max_float){
					max_float=last_layer->y.d[no][x_aixs*a+i][y_aixs*b+j];
				}
			}
		}

//	#ifdef	_DEBUG_IN_MP_
//		printf("return=%f\n",max_float);
//	#endif
		return max_float;
	}

	void init_1(Convolutional_Layer *last,int a_core,int b_core);
	void init_2(void *next,bool fcl);

	void calculate_y(){
		for (int i=0;i<num;i++){
			for (int j=0;j<m;j++){
				for (int k=0;k<n;k++){
					d.d[i][j][k]=beta.d[i][j][k]*max(i,j,k);
			#ifdef	_DEBUG_IN_MP_
				printf("D:[%d][%d][%d]=%f\n",i,j,k,d.d[i][j][k]);
			#endif
					y.d[i][j][k]=sigmoid::y(d.d[i][j][k]+bias.d[i][j][k]);

			#ifdef	_DEBUG_IN_MP_
				printf("y:[%d][%d][%d]=%f\n",i,j,k,y.d[i][j][k]);
			#endif
				}
			}
		}
	}

	void calculate_dleta();

	void change_weight(float eta){
		for (int i=0;i<num;i++){
			for (int j=0;j<m;j++){
				for (int k=0;k<n;k++){
//					std::cout<<i<<','<<j <<','<<k<<std::endl;
					beta.d[i][j][k]+=eta*dleta.d[i][j][k];
					bias.d[i][j][k]+=eta*dleta.d[i][j][k];
				}
			}
		}
	}
};

class Fully_Connected_Layer{
public:
	void *w;
	array bias;
	int last_m,last_n;
	int last_num;
	int next_num;
	int num;
	float *y;
	void *last_layer;
	Fully_Connected_Layer *next_layer;
	bool max_pooling;
	bool end;
	array dleta;

	void init_1(void *last,int n,bool mpl){
		num=n;
		last_layer=last;
		max_pooling=mpl;
		y=new float[num];
		if (max_pooling==true){
			w=new cube;
			last_m=((Max_Pooling_Layer*)last_layer)->m;
			last_n=((Max_Pooling_Layer*)last_layer)->n;
			last_num=((Max_Pooling_Layer*)last_layer)->num;
			((tube*)w)->init(last_num,last_m,last_n,n,NULL);
		}else{
			w=new mat;
			last_num=((Fully_Connected_Layer*)last_layer)->num;
			((mat*)w)->init(((Fully_Connected_Layer*)last_layer)->num,n,NULL);
		}
		dleta.init(num);
		bias.init(num);
	#ifdef _DEBUG_INIT_
		std::cout<<"--\nFully_Connected_Layer Init Stage 1:"<<std::endl;
		std::cout<<"n,last_num,max_pooling"<<std::endl;
		std::cout<<n<<','<<last_num<<','<<max_pooling<<std::endl;
	#endif
	}

	void init_2(Fully_Connected_Layer *next){
		if (next==NULL){
			next_layer=NULL;
			end=true;
		}else{
			next_layer=next;
			next_num=next_layer->num;
			end=false;
		}
	#ifdef _DEBUG_INIT_
		std::cout<<"--\nFully_Connected_Layer Init Stage 2:"<<std::endl;
		std::cout<<"end,next_num"<<std::endl;
		std::cout<<end<<','<<next_num<<std::endl;
	#endif
	}

	void calculate_y(){
		if (max_pooling==true){
			for (int i=0;i<num;i++){
				float sum=0;
				for (int j=0;j<last_num;j++){
					for(int k=0;k<last_m;k++){
						for(int l=0;l<last_n;l++){
							sum=sum+((tube*)w)->d[j][k][l][i]*((Max_Pooling_Layer*)last_layer)->y.d[j][k][l];
						}
					}
				}
				y[i]=sigmoid::y(sum+bias.d[i]);
			}
		}else{
			for (int i=0;i<num;i++){
				float sum=0;
				for (int j=0;j<last_num;j++){
					sum=sum+((mat*)w)->d[j][i]*((Fully_Connected_Layer*)last_layer)->y[j];
				}
				y[i]=sigmoid::y(sum+bias.d[i]);
			}
		}
	}

	void calculate_dleta(float *d){
		if(end==true){
			for (int i=0;i<num;i++){
				dleta.d[i]=sigmoid::df(y[i])*(d[i]-y[i]);
			}
		}else{
//			std::cout<<"max"<<std::endl;
			for (int i=0;i<num;i++){
				float sum=0;
				for (int j=0;j<next_num;j++) {
					sum=sum+((mat*)((Fully_Connected_Layer*)next_layer)->w)->d[i][j]*((Fully_Connected_Layer*)next_layer)->dleta.d[j];
				}
				dleta.d[i]=sum*sigmoid::df(y[i]);
			}
		}
	}

	void change_weight(float eta){
		if(max_pooling==false){
//			std::cout<<"wtf"<<std::endl;
			for (int i=0;i<num;i++){
				for (int j=0;j<last_num;j++) {
					((mat*)w)->d[j][i]+=eta*dleta.d[i]*((Fully_Connected_Layer*)last_layer)->y[j];
				}
				bias.d[i]+=eta*dleta.d[i];
			}
		}else{
//			std::cout<<"maxxxx"<<std::endl;
			for (int i=0;i<num;i++){
//				std::cout<<i<<std::endl;
				for (int j=0;j<last_num;j++){
					for(int k=0;k<last_m;k++){
						for(int l=0;l<last_n;l++){
							((tube*)w)->d[j][k][l][i]+=eta*dleta.d[i]*((Max_Pooling_Layer*)last_layer)->y.d[j][k][l];
						}
					}
				}
				bias.d[i]+=eta*dleta.d[i];
			}
		}
	}
};

//层类相关函数

void Convolutional_Layer::init_1(void *last,int num_pics,int a_core,int b_core,bool start){
	s=start;
	num=num_pics;
	a=a_core;
	b=b_core;
	last_layer=last;
	if(start==true){
		last_num=((Input_Layer*)last_layer)->num;
		last_n=((Input_Layer*)last_layer)->n;
		last_m=((Input_Layer*)last_layer)->m;
	}else{
		last_num=((Max_Pooling_Layer*)last_layer)->num;
		last_n=((Max_Pooling_Layer*)last_layer)->n;
		last_m=((Max_Pooling_Layer*)last_layer)->m;
	}
	m=last_m-a+1;
	n=last_n-a+1;
	w.init(num,last_num,a,b,NULL);
	bias.init(num,m,n);
	y.init(num,m,n);
	dleta.init(num,m,n);
	#ifdef _DEBUG_INIT_
		std::cout<<"--\nConvolutional_Layer Init Stage 1:"<<std::endl;
		std::cout<<"s,num,a,b,m,n"<<std::endl;
		std::cout<<s<<','<<num<<','<<a<<','<<b<<','<<m<<','<<n<<std::endl;
	#endif
}

void Convolutional_Layer::init_2(Max_Pooling_Layer *next){
	next_layer=next;
	next_num=next->num;
	next_a=next->a;
	next_b=next->b;
	#ifdef _DEBUG_INIT_
		std::cout<<"--\nConvolutional_Layer Init Stage 2:"<<std::endl;
		std::cout<<"next_num,next_a,next_b"<<std::endl;
		std::cout<<next_num<<','<<next_a<<','<<next_b<<std::endl;
	#endif
}

float Convolutional_Layer::core(int x_aixs,int y_aixs,int num_source,int num_output){
		float sum=0;
		if(s==true){
			for(int i=0;i<a;i++){
				for(int j=0;j<b;j++){
					sum=sum+w.d[num_output][num_source][i][j]*((Input_Layer*)last_layer)->y.d[num_source][x_aixs+i][y_aixs+j];
				}
			}
		}else{
			for(int i=0;i<a;i++){
				for(int j=0;j<b;j++){
					sum=sum+w.d[num_output][num_source][i][j]*((Max_Pooling_Layer*)last_layer)->y.d[num_source][x_aixs+i][y_aixs+j];
				}
			}
		}
		return sum;
	}

void Max_Pooling_Layer::init_1(Convolutional_Layer *last,int a_core,int b_core){
	num=last->num;
	a=a_core;
	b=b_core;
	m=last->m/a;
	n=last->n/b;
	last_layer=last;
	beta.init(num,m,n,NULL);
	bias.init(num,m,n,NULL);
	y.init(num,m,n);
	d.init(num,m,n);
	dleta.init(num,m,n);
	#ifdef _DEBUG_INIT_
		std::cout<<"--\nMax_Pooling_Layer Init Stage 1:"<<std::endl;
		std::cout<<"num,m,n"<<std::endl;
		std::cout<<num<<','<<m<<','<<n<<std::endl;
	#endif
}

void Max_Pooling_Layer::init_2(void *next,bool fcl){
	Fully_Connected=fcl;
	next_layer=next;
	if (Fully_Connected==true){
		next_num=((Fully_Connected_Layer*)next_layer)->num;
	}else{
		next_num=((Convolutional_Layer*)next_layer)->num;
		next_m=((Convolutional_Layer*)next_layer)->m;
		next_n=((Convolutional_Layer*)next_layer)->n;
		next_a=((Convolutional_Layer*)next_layer)->a;
		next_b=((Convolutional_Layer*)next_layer)->b;
	}
	#ifdef _DEBUG_INIT_
		std::cout<<"--\nMax_Pooling_Layer Init Stage 2:"<<std::endl;
		std::cout<<"Fully_Connected"<<std::endl;
		std::cout<<Fully_Connected<<std::endl;
	#endif

}

void Max_Pooling_Layer::calculate_dleta(){
	if (Fully_Connected==true){
		for (int i=0;i<num;i++){
			for (int j=0;j<m;j++){
				for (int k=0;k<n;k++){
					float sum=0;
					for (int l=0;l<next_num;l++){
						sum=sum+((Fully_Connected_Layer*)next_layer)->dleta.d[l]*((tube*)((Fully_Connected_Layer*)next_layer)->w)->d[i][j][k][l];
					}
					dleta.d[i][j][k]=sigmoid::df(y.d[i][j][k])*sum;
				}
			}
		}
	}else{
//		std::cout<<num<<','<<m<<','<<n<<','<<next_num<<','<<next_a<<','<<next_b<<std::endl;
		for (int i=0;i<num;i++){
			for (int j=0;j<m;j++){
				for (int k=0;k<n;k++){
					float sum=0;
					float x_s,x_e,y_s,y_e;
					if(j<next_a){
						x_s=0;
						x_e=j+1;
					}else if(j>(m-next_a)){
						x_s=next_a-(m-j);
						x_e=next_a;
					}else{
						x_s=0;
						x_e=next_a;
					}
					if(k<next_b){
						y_s=0;
						y_e=k+1;
					}else if(k>(n-next_b)){
						y_s=next_b-(n-k);
						y_e=next_b;
					}else{
						y_s=0;
						y_e=next_b;
					}
//					std::cout<<x_s<<'|'<<x_e<<'|'<<y_s<<'|'<<y_e<<std::endl;
					for (int l_1=0;l_1<next_num;l_1++){
						for (int l_2=x_s;l_2<x_e;l_2++){
							for (int l_3=y_s;l_3<y_e;l_3++){
//								std::cout<<i<<','<<j<<','<<k<<','<<l_1<<','<<l_2<<','<<l_3<<std::endl;
								sum=sum+((Convolutional_Layer*)next_layer)->dleta.d[l_1][j-l_2][k-l_3]*((Convolutional_Layer*)next_layer)->w.d[l_1][i][l_2][l_3];
							}
						}
					}
					dleta.d[i][j][k]=sigmoid::df(y.d[i][j][k])*sum;
				}
			}
		}
	}
}

void Convolutional_Layer::calculate_dleta(){
	for (int i=0;i<num;i++){
		for (int j=0;j<m;j++){
			for (int k=0;k<n;k++){
				dleta.d[i][j][k]=next_layer->beta.d[i][j/next_a][k/next_a]*sigmoid::df(y.d[i][j][k])*next_layer->dleta.d[i][j/next_a][k/next_a];
			}
		}
	}
}

void Convolutional_Layer::change_weight(float eta){
	if(s==true){
		for(int i=0;i<num;i++){
			for(int j=0;j<a;j++){
				for(int k=0;k<b;k++){
					for (int l_1=0;l_1<last_num;l_1++){
						float sum=0;
						for (int l_2=0;l_2<(last_m-a);l_2++){
							for (int l_3=0;l_3<(last_n-b);l_3++){
								sum=sum+((Input_Layer*)last_layer)->y.d[l_1][l_2+j][l_3+k]*w.d[i][l_1][j][k]*dleta.d[i][j][k];
							}
						}
						w.d[i][l_1][j][k] +=eta*(sum/((last_m-a)*(last_n-b)));
					}
				}
			}
			for(int j=0;j<m;j++){
				for(int k=0;k<n;k++){
					bias.d[i][j][k]+=eta*dleta.d[i][j][k];
				}
			}
		}
	}else{
		for(int i=0;i<num;i++){
			for(int j=0;j<a;j++){
				for(int k=0;k<b;k++){
					for (int l_1=0;l_1<last_num;l_1++){
						float sum=0;
						for (int l_2=0;l_2<(last_m-a);l_2++){
							for (int l_3=0;l_3<(last_n-b);l_3++){
								sum=sum+((Max_Pooling_Layer*)last_layer)->y.d[l_1][l_2+j][l_3+k]*w.d[i][l_1][j][k]*dleta.d[i][j][k];
							}
						}
						w.d[i][l_1][j][k] +=eta*(sum/((last_m-a)*(last_n-b)));
					}
				}
			}
			for(int j=0;j<m;j++){
				for(int k=0;k<n;k++){
					bias.d[i][j][k]+=eta*dleta.d[i][j][k];
				}
			}
		}
	}
}


class Convolutional_Neural_Network{
public:
	Input_Layer INPUT;
	Convolutional_Layer C_1;
	Max_Pooling_Layer MP_2;
	Convolutional_Layer C_3;
	Max_Pooling_Layer MP_4;
	Fully_Connected_Layer FC_7;
	Fully_Connected_Layer FC_8;
	Fully_Connected_Layer FC_9;

//	void init(){
//		INPUT.init(1,28,28);
//		C_1.init_1((void*)&INPUT,8,3,3,true);
//		MP_2.init_1(&C_1,2,2);
//		C_1.init_2(&MP_2);
//		C_3.init_1((void*)&MP_2,16,2,2,false);
//		MP_2.init_2(&C_3,false);
//		MP_4.init_1(&C_3,2,2);
//		C_3.init_2(&MP_4);
//		C_5.init_1((void*)&MP_4,32,3,3,false);
//		MP_4.init_2(&C_5,false);
//		MP_6.init_1(&C_5,2,2);
//		C_5.init_2(&MP_6);
//		FC_7.init_1(&MP_6,128,true);
//		MP_6.init_2(&FC_7,true);
//		FC_8.init_1(&FC_7,10,false);
//		FC_7.init_2(&FC_8);
//		FC_9.init_1(&FC_8,10,false);
//		FC_8.init_2(&FC_9);
//		FC_9.init_2(NULL);
//	}

	void init(){
		INPUT.init(1,28,28);
		C_1.init_1((void*)&INPUT,8,5,5,true);
		MP_2.init_1(&C_1,2,2);
		C_1.init_2(&MP_2);
		C_3.init_1((void*)&MP_2,16,3,3,false);
		MP_2.init_2(&C_3,false);
		MP_4.init_1(&C_3,2,2);
		C_3.init_2(&MP_4);
		FC_7.init_1(&MP_4,128,true);
		MP_4.init_2(&FC_7,true);
		FC_8.init_1(&FC_7,10,false);
		FC_7.init_2(&FC_8);
		FC_9.init_1(&FC_8,10,false);
		FC_8.init_2(&FC_9);
		FC_9.init_2(NULL);
	}

	void calculate(){
	#ifdef	_DEBUG_IN_MP_
		for(int i=0;i<INPUT.num;i++){
			for(int j=0;j<INPUT.m;j++){
				for(int k=0;k<INPUT.n;k++){
					std::cout<<INPUT.y.d[i][j][k];
				}
				std::cout<<std::endl;
			}
			std::cout<<"==================="<<std::endl;
		}
	#endif
	#ifdef	_DEBUG_Y_
		std::cout<<"======\nY\n======\nC_1"<<std::endl;
	#endif
		C_1.calculate_y();
	#ifdef	_DEBUG_IN_MP_
		for(int i=0;i<C_1.num;i++){
			for(int j=0;j<C_1.m;j++){
				for(int k=0;k<C_1.n;k++){
					std::cout<<C_1.y.d[i][j][k];
				}
				std::cout<<std::endl;
			}
			std::cout<<"==================="<<std::endl;
		}
	#endif
	#ifdef	_DEBUG_Y_
		std::cout<<"MP_2"<<std::endl;
	#endif
		MP_2.calculate_y();
	#ifdef	_DEBUG_IN_MP_
		for(int i=0;i<MP_2.num;i++){
			for(int j=0;j<MP_2.m;j++){
				for(int k=0;k<MP_2.n;k++){
					std::cout<<MP_2.y.d[i][j][k];
				}
				std::cout<<std::endl;
			}
			std::cout<<"==================="<<std::endl;
		}
	#endif
	#ifdef	_DEBUG_Y_
		std::cout<<"C_3"<<std::endl;
	#endif
		C_3.calculate_y();
	#ifdef	_DEBUG_Y_
		std::cout<<"MP_4"<<std::endl;
	#endif
		MP_4.calculate_y();
	#ifdef	_DEBUG_Y_
		std::cout<<"C_5"<<std::endl;
	#endif
//		C_5.calculate_y();
	#ifdef	_DEBUG_Y_
		std::cout<<"MP_6"<<std::endl;
	#endif
//		MP_6.calculate_y();
	#ifdef	_DEBUG_Y_
		std::cout<<"FC_7"<<std::endl;
	#endif
		FC_7.calculate_y();
	#ifdef	_DEBUG_Y_
		std::cout<<"FC_8"<<std::endl;
	#endif
		FC_8.calculate_y();
	#ifdef	_DEBUG_Y_
		std::cout<<"FC_9"<<std::endl;
	#endif
		FC_9.calculate_y();
	#ifdef	_DEBUG_Y_
		std::cout<<"END_Y"<<std::endl;
	#endif
	}

	void train(float *y,float eta_c,float eta_mp,float eta_fc){
		calculate();
	#ifdef	_DEBUG_T_
		std::cout<<"FC_9"<<std::endl;
	#endif
		FC_9.calculate_dleta(y);
	#ifdef	_DEBUG_T_
		std::cout<<"FC_8"<<std::endl;
	#endif
		FC_8.calculate_dleta(NULL);
	#ifdef	_DEBUG_T_
		std::cout<<"FC_7"<<std::endl;
	#endif
		FC_7.calculate_dleta(NULL);
	#ifdef	_DEBUG_T_
		std::cout<<"MP_6"<<std::endl;
	#endif
//		MP_6.calculate_dleta();
	#ifdef	_DEBUG_T_
		std::cout<<"C_5"<<std::endl;
	#endif
//		C_5.calculate_dleta();
	#ifdef	_DEBUG_T_
		std::cout<<"MP_4"<<std::endl;
	#endif
		MP_4.calculate_dleta();
	#ifdef	_DEBUG_T_
		std::cout<<"C_3"<<std::endl;
	#endif
		C_3.calculate_dleta();
	#ifdef	_DEBUG_T_
		std::cout<<"MP_2"<<std::endl;
	#endif
		MP_2.calculate_dleta();
	#ifdef	_DEBUG_T_
		std::cout<<"C_1"<<std::endl;
	#endif
		C_1.calculate_dleta();
	#ifdef	_DEBUG_T_
		std::cout<<"=========================================="<<std::endl;
	#endif
	#ifdef	_DEBUG_T_
		std::cout<<"FC_9"<<std::endl;
	#endif
		FC_9.change_weight(eta_fc);
	#ifdef	_DEBUG_T_
		std::cout<<"FC_8"<<std::endl;
	#endif
		FC_8.change_weight(eta_fc);
	#ifdef	_DEBUG_T_
		std::cout<<"FC_7"<<std::endl;
	#endif
		FC_7.change_weight(eta_fc);
	#ifdef	_DEBUG_T_
		std::cout<<"MP_6"<<std::endl;
	#endif
//		MP_6.change_weight(eta);
	#ifdef	_DEBUG_T_
		std::cout<<"C_5"<<std::endl;
	#endif
//		C_5.change_weight(eta);
	#ifdef	_DEBUG_T_
		std::cout<<"MP_4"<<std::endl;
	#endif
		MP_4.change_weight(eta_mp);
	#ifdef	_DEBUG_T_
		std::cout<<"C_3"<<std::endl;
	#endif
		C_3.change_weight(eta_c);
	#ifdef	_DEBUG_T_
		std::cout<<"MP_2"<<std::endl;
	#endif
		MP_2.change_weight(eta_mp);
	#ifdef	_DEBUG_T_
		std::cout<<"C_1"<<std::endl;
	#endif
		C_1.change_weight(eta_c);
	}
};
