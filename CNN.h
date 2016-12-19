#include <stdlib.h>
#include <math.h>

//ç»«å©šî•©ç€¹æ°«ç®Ÿ

class tube;
class cube;
class mat;
class array;
class Input_Layer;
class Convolutional_Layer;
class Max_Pooling_Layer;
class Fully_Connected_Layer;

//é—…å¿”æº€é‘èŠ¥æšŸ

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

// æˆæ’³åš­é‘èŠ¥æšŸ

namespace sigmoid{
	float y(float x){
		return 1/(1+exp(-x));
	}

	float df(float y){
		return y*(1-y);
	}
}

//éç‰ˆåµç€›æ¨ºåç»«?

//4 aixs
class tube{
public:
	float ****d;
	int a_1,a_2,a_3,a_4;
	void init(int x,int y,int z,int a){
		a_1=x;
		a_2=y;
		a_3=z;
		a_4=a;
		d=new float***[x];
		for (int i=0;i<x;i++){
			d[i]=new float**[y];
			for (int j=0;j<y;j++){
				d[i][j]=new float*[z];
				for(int k=0;k<z;k++){
					d[i][j][k]=new float[a];
					for(int l=0;l<a;l++){
						d[i][j][k][l]=0;
					}
				}
			}
		}
	}

	void init(int x,int y,int z,int a,void *r_init){
		a_1=x;
		a_2=y;
		a_3=z;
		a_4=a;
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
	
	void add(tube *source){
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
				for(int k=0;k<a_3;k++){
					for(int l=0;l<a_4;l++){
						d[i][j][k][l]+=source->d[i][j][k][l];
					}
				}
			}
		}
	}
	
	void add(tube *source,float eta){
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
				for(int k=0;k<a_3;k++){
					for(int l=0;l<a_4;l++){
						d[i][j][k][l]+=eta*source->d[i][j][k][l];
					}
				}
			}
		}
	}
	
	void copy(tube *source){
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
				for(int k=0;k<a_3;k++){
					for(int l=0;l<a_4;l++){
						d[i][j][k][l]=source->d[i][j][k][l];
					}
				}
			}
		}
	}
	
	void reset(){
		
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
				for(int k=0;k<a_3;k++){
					for(int l=0;l<a_4;l++){
						d[i][j][k][l]=0;
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
	int a_1,a_2,a_3;

	void init(int x,int y,int z){
		a_1=x;
		a_2=y;
		a_3=z;
		d=new float**[x];
		for (int i=0;i<x;i++){
			d[i]=new float*[y];
			for (int j=0;j<y;j++){
				d[i][j]=new float[z];
				for(int k=0;k<z;k++){
					d[i][j][k]=0;
				}
			}
		}
	}

	void init(int x,int y,int z,void *r_init){
		a_1=x;
		a_2=y;
		a_3=z;
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
	
	void add(cube *source){
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
				for(int k=0;k<a_3;k++){
					d[i][j][k]+=source->d[i][j][k];
				}
			}
		}
	}
	
	void add(cube *source,float eta){
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
				for(int k=0;k<a_3;k++){
					d[i][j][k]+=eta*source->d[i][j][k];
				}
			}
		}
	}
	
	void copy(cube *source){
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
				for(int k=0;k<a_3;k++){
					d[i][j][k]=source->d[i][j][k];
				}
			}
		}
	}
	
	void reset(){
		
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
				for(int k=0;k<a_3;k++){
					d[i][j][k]=0;
				}
			}
		}
	}
};

//2 aixs
class mat{
public:
	float **d;
	int a_1,a_2;

	void init(int x,int y){
		a_1=x;
		a_2=y;
		d=new float*[x];
		for (int i=0;i<x;i++){
			d[i]=new float[y];
			for (int j=0;j<y;j++){
				d[i][j]=0;
			}
		}
	}

	void init(int x,int y,void *r_init){
		a_1=x;
		a_2=y;
		d=new float*[x];
		for (int i=0;i<x;i++){
			d[i]=new float[y];
			for (int j=0;j<y;j++){
				d[i][j]=R();
			}
		}
	}
	
	void add(mat *source){
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
					d[i][j]+=source->d[i][j];
			}
		}
	}
	
	void add(mat *source,float eta){
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
					d[i][j]+=eta*source->d[i][j];
			}
		}
	}
	
	void copy(mat *source){
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
					d[i][j]=source->d[i][j];
			}
		}
	}
	
	void reset(){
		
		for (int i=0;i<a_1;i++){
			for (int j=0;j<a_2;j++){
					d[i][j]=0;
			}
		}
	}
};

class array{
public:
	float *d;
	int a_1;

	void init(int num){
		a_1=num;
		d=new float[num];
		for (int i=0;i<num;i++){
			d[i]=0;
		}
	}
	
	void init(int num,void *r_init){
		a_1=num;
		d=new float[num];
		for (int i=0;i<num;i++){
			d[i]=R();
		}
	}
	
	void add(array *source){
		for (int i=0;i<a_1;i++){
				d[i]+=source->d[i];
		}
	}
	
	void add(array *source,float eta){
		for (int i=0;i<a_1;i++){
				d[i]+=eta*source->d[i];
		}
	}
	
	void copy(array *source){
		for (int i=0;i<a_1;i++){
				d[i]=source->d[i];
		}
	}
	
	void reset(){
		
		for (int i=0;i<a_1;i++){
				d[i]=0;
		}
	}
};

//çžå‚œè¢«

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
	tube d_w;
	int m,n;
	int a,b;
	cube y;
	cube dleta;
	cube bias;
	cube d_bias;
	int num;
	void *last_layer;
	int last_m,last_n;
	int last_num,last_use;
	Max_Pooling_Layer *next_layer;
	int next_num;
	int next_a,next_b;
	bool s;

	float core(int x_aixs,int y_aixs,int num_source,int num_output,int index);

	void init_1(void *last,int num_pics,int a_core,int b_core,bool start,int use);
	void init_2(Max_Pooling_Layer *next);

	void calculate_y(){
		for(int i=0;i<num;i++){
			for(int j=0;j<m;j++){
				for(int k=0;k<n;k++){
					float sum=0;
					for(int l=0;l<last_use;l++){
						sum=sum+core(j,k,(i+l)%last_num,i,l);
					}
					y.d[i][j][k]=sigmoid::y(sum+bias.d[i][j][k]);
				}
			}
		}
	}

	void calculate_dleta();
	void calculate_d_w();

	void change_weight(float eta);
	void change_weight(Convolutional_Layer *source,float eta);
	void copy_weight(Convolutional_Layer *source);
};

class Max_Pooling_Layer{
public:
	cube beta,bias;
	cube d_beta,d_bias;
	Convolutional_Layer *last_layer;
	int m,n;
	int a,b;
	int num;
	void *next_layer;
	int next_m,next_n;
	int next_a,next_b;
	int next_num;
	int next_use;
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
	
	void calculate_d_w(){
		for (int i=0;i<num;i++){
			for (int j=0;j<m;j++){
				for (int k=0;k<n;k++){
					d_beta.d[i][j][k]+=dleta.d[i][j][k];
					d_bias.d[i][j][k]+=dleta.d[i][j][k];
				}
			}
		}
	}

	void change_weight(float eta){
		beta.add(&d_beta,eta);
		bias.add(&d_bias,eta);
		d_beta.reset();
		d_bias.reset();
	}

	void change_weight(Max_Pooling_Layer *source,float eta){
		beta.add(&(source->d_beta),eta);
		bias.add(&(source->d_bias),eta);
		source->d_beta.reset();
		source->d_bias.reset();
	}

	void copy_weight(Max_Pooling_Layer *source){
		beta.copy(&(source->beta));
		bias.copy(&(source->bias));
	}
};

class Fully_Connected_Layer{
public:
	void *w;
	void *d_w;
	array bias;
	array d_bias;
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
			d_w=new cube;
			last_m=((Max_Pooling_Layer*)last_layer)->m;
			last_n=((Max_Pooling_Layer*)last_layer)->n;
			last_num=((Max_Pooling_Layer*)last_layer)->num;
			((tube*)w)->init(last_num,last_m,last_n,n,NULL);
			((tube*)d_w)->init(last_num,last_m,last_n,n,NULL);
		}else{
			w=new mat;
			d_w=new mat;
			last_num=((Fully_Connected_Layer*)last_layer)->num;
			((mat*)w)->init(((Fully_Connected_Layer*)last_layer)->num,n,NULL);
			((mat*)d_w)->init(((Fully_Connected_Layer*)last_layer)->num,n,NULL);
		}
		dleta.init(num);
		bias.init(num);
		d_bias.init(num);
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
	
	void calculate_d_w(){
		if(max_pooling==false){
			for (int i=0;i<num;i++){
				for (int j=0;j<last_num;j++) {
					((mat*)d_w)->d[j][i]+=dleta.d[i]*((Fully_Connected_Layer*)last_layer)->y[j];
				}
				d_bias.d[i]+=dleta.d[i];
			}
		}else{
			for (int i=0;i<num;i++){
				for (int j=0;j<last_num;j++){
					for(int k=0;k<last_m;k++){
						for(int l=0;l<last_n;l++){
							((tube*)d_w)->d[j][k][l][i]+=dleta.d[i]*((Max_Pooling_Layer*)last_layer)->y.d[j][k][l];
						}
					}
				}
				d_bias.d[i]+=dleta.d[i];
			}
		}
	}

	void change_weight(float eta){
		if(max_pooling==false){
			((mat*)w)->add(((mat*)d_w),eta);
			bias.add(&d_bias,eta);
			((mat*)d_w)->reset();
			d_bias.reset();
		}else{
			((tube*)w)->add(((tube*)d_w),eta);
			bias.add(&d_bias,eta);
			((tube*)d_w)->reset();
			d_bias.reset();
		}
	
	}

	void change_weight(Fully_Connected_Layer *source,float eta){
		if(max_pooling==false){
			((mat*)w)->add(((mat*)source->d_w),eta);
			bias.add(&(source->d_bias),eta);
			((mat*)source->d_w)->reset();
			source->d_bias.reset();
		}else{
			((tube*)w)->add(((tube*)source->d_w),eta);
			bias.add(&(source->d_bias),eta);
			((tube*)source->d_w)->reset();
			source->d_bias.reset();
		}
	}

	void copy_weight(Fully_Connected_Layer *source){
		if(max_pooling==false){
			((mat*)w)->copy(((mat*)source->w));
			bias.copy(&(source->bias));
		}else{
			((tube*)w)->copy(((tube*)source->w));
			bias.copy(&(source->bias));
		}
	}
};

//çžå‚œè¢«é©ç¨¿å§é‘èŠ¥æšŸ

void Convolutional_Layer::init_1(void *last,int num_pics,int a_core,int b_core,bool start,int use){
	s=start;
	num=num_pics;
	a=a_core;
	b=b_core;
	last_layer=last;
	last_use=use;
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
	w.init(num,last_use,a,b,NULL);
	bias.init(num,m,n);
	d_w.init(num,last_use,a,b);
	d_bias.init(num,m,n);
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

float Convolutional_Layer::core(int x_aixs,int y_aixs,int num_source,int num_output,int index){
		float sum=0;
		if(s==true){
			for(int i=0;i<a;i++){
				for(int j=0;j<b;j++){
					sum=sum+w.d[num_output][index][i][j]*((Input_Layer*)last_layer)->y.d[num_source][x_aixs+i][y_aixs+j];
				}
			}
		}else{
			for(int i=0;i<a;i++){
				for(int j=0;j<b;j++){
					sum=sum+w.d[num_output][index][i][j]*((Max_Pooling_Layer*)last_layer)->y.d[num_source][x_aixs+i][y_aixs+j];
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
	bias.init(num,m,n);
	d_beta.init(num,m,n);
	d_bias.init(num,m,n);
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
		next_use=((Convolutional_Layer*)next_layer)->last_use;
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
		int l_num=0;
		int r_index=0;
		if((next_num%num)>=next_use){
			l_num=((next_num/num)+1)*next_use;
		}else{
			l_num=(next_num/num)*next_use+next_num%num;
		}
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
					for (int l_1=0;l_1<l_num;l_1++){
						int index=0;
						int temp=(next_use-l_1-1);
						if(i<next_use){
							index=((next_use-i-1+l_1)/(next_use))*(num)+(next_use-i-1+l_1)%(next_use)-next_use+i+1;
							r_index=next_use-((l_1+next_use-i-1)%next_use)-1;
						}else{
							index=((l_1)/(next_use))*(num)+(l_1)%(next_use)-next_use+i+1;
							r_index=next_use-((l_1)%next_use)-1;
						}
						for (int l_2=x_s;l_2<x_e;l_2++){
							for (int l_3=y_s;l_3<y_e;l_3++){
//								std::cout<<"hhh"<<index<<','<<next_use<<','<<r_index<<','<<l_num<<','<<l_1<<','<<i<<std::endl;
								sum=sum+((Convolutional_Layer*)next_layer)->dleta.d[index][j-l_2][k-l_3]*((Convolutional_Layer*)next_layer)->w.d[index][r_index][l_2][l_3];
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

void Convolutional_Layer::calculate_d_w(){
	if(s==true){
		for(int i=0;i<num;i++){
			for(int j=0;j<a;j++){
				for(int k=0;k<b;k++){
					for (int l_1=0;l_1<last_use;l_1++){
						float sum=0;
						for (int l_2=0;l_2<(last_m-a);l_2++){
							for (int l_3=0;l_3<(last_n-b);l_3++){
								sum=sum+((Input_Layer*)last_layer)->y.d[(i+l_1)%last_num][l_2+j][l_3+k]*w.d[i][l_1][j][k]*dleta.d[i][j][k];
							}
						}
						d_w.d[i][l_1][j][k] +=(sum/((last_m-a)*(last_n-b)));
					}
				}
			}
			for(int j=0;j<m;j++){
				for(int k=0;k<n;k++){
					d_bias.d[i][j][k]+=dleta.d[i][j][k];
				}
			}
		}
	}else{
		for(int i=0;i<num;i++){
			for(int j=0;j<a;j++){
				for(int k=0;k<b;k++){
					for (int l_1=0;l_1<last_use;l_1++){
						float sum=0;
						for (int l_2=0;l_2<(last_m-a);l_2++){
							for (int l_3=0;l_3<(last_n-b);l_3++){
								sum=sum+((Max_Pooling_Layer*)last_layer)->y.d[(i+l_1)%last_num][l_2+j][l_3+k]*w.d[i][l_1][j][k]*dleta.d[i][j][k];
							}
						}
						d_w.d[i][l_1][j][k] +=(sum/((last_m-a)*(last_n-b)));
					}
				}
			}
			for(int j=0;j<m;j++){
				for(int k=0;k<n;k++){
					d_bias.d[i][j][k]+=dleta.d[i][j][k];
				}
			}
		}
	}
}

void Convolutional_Layer::change_weight(float eta){
	w.add(&d_w,eta);
	bias.add(&d_bias,eta);
	d_w.reset();
	d_bias.reset();
}

void Convolutional_Layer::change_weight(Convolutional_Layer *source,float eta){
	w.add(&(source->d_w),eta);
	bias.add(&(source->d_bias),eta);
	(source->d_w).reset();
	(source->d_bias).reset();
}

void Convolutional_Layer::copy_weight(Convolutional_Layer *source){
	w.copy(&(source->w));
	bias.copy(&(source->bias));
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
		C_1.init_1((void*)&INPUT,8,5,5,true,1);
		MP_2.init_1(&C_1,2,2);
		C_1.init_2(&MP_2);
		C_3.init_1((void*)&MP_2,16,3,3,false,3);
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

	void train(float *y){
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
		FC_9.calculate_d_w();
	#ifdef	_DEBUG_T_
		std::cout<<"FC_8"<<std::endl;
	#endif
		FC_8.calculate_d_w();
	#ifdef	_DEBUG_T_
		std::cout<<"FC_7"<<std::endl;
	#endif
		FC_7.calculate_d_w();
	#ifdef	_DEBUG_T_
		std::cout<<"MP_6"<<std::endl;
	#endif
//		MP_6.calculate_d_w();
	#ifdef	_DEBUG_T_
		std::cout<<"C_5"<<std::endl;
	#endif
//		C_5.calculate_d_w();
	#ifdef	_DEBUG_T_
		std::cout<<"MP_4"<<std::endl;
	#endif
		MP_4.calculate_d_w();
	#ifdef	_DEBUG_T_
		std::cout<<"C_3"<<std::endl;
	#endif
		C_3.calculate_d_w();
	#ifdef	_DEBUG_T_
		std::cout<<"MP_2"<<std::endl;
	#endif
		MP_2.calculate_d_w();
	#ifdef	_DEBUG_T_
		std::cout<<"C_1"<<std::endl;
	#endif
		C_1.calculate_d_w();
	#ifdef	_DEBUG_T_
		std::cout<<"=========================================="<<std::endl;
	#endif
	}
	
	void change_weight(float eta)
	{
	#ifdef	_DEBUG_T_
		std::cout<<"FC_9"<<std::endl;
	#endif
		FC_9.change_weight(eta);
	#ifdef	_DEBUG_T_
		std::cout<<"FC_8"<<std::endl;
	#endif
		FC_8.change_weight(eta);
	#ifdef	_DEBUG_T_
		std::cout<<"FC_7"<<std::endl;
	#endif
		FC_7.change_weight(eta);
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
		MP_4.change_weight(eta);
	#ifdef	_DEBUG_T_
		std::cout<<"C_3"<<std::endl;
	#endif
		C_3.change_weight(eta);
	#ifdef	_DEBUG_T_
		std::cout<<"MP_2"<<std::endl;
	#endif
		MP_2.change_weight(eta);
	#ifdef	_DEBUG_T_
		std::cout<<"C_1"<<std::endl;
	#endif
		C_1.change_weight(eta);
	}
	
	void change_weight(Convolutional_Neural_Network *source,float eta)
	{
	#ifdef	_DEBUG_T_
		std::cout<<"FC_9"<<std::endl;
	#endif
		FC_9.change_weight(&(source->FC_9),eta);
	#ifdef	_DEBUG_T_
		std::cout<<"FC_8"<<std::endl;
	#endif
		FC_8.change_weight(&(source->FC_8),eta);
	#ifdef	_DEBUG_T_
		std::cout<<"FC_7"<<std::endl;
	#endif
		FC_7.change_weight(&(source->FC_7),eta);
	#ifdef	_DEBUG_T_
		std::cout<<"MP_6"<<std::endl;
	#endif
//		MP_6.change_weight(&(source->MP_6),eta);
	#ifdef	_DEBUG_T_
		std::cout<<"C_5"<<std::endl;
	#endif
//		C_5.change_weight(&(source->C_5),eta);
	#ifdef	_DEBUG_T_
		std::cout<<"MP_4"<<std::endl;
	#endif
		MP_4.change_weight(&(source->MP_4),eta);
	#ifdef	_DEBUG_T_
		std::cout<<"C_3"<<std::endl;
	#endif
		C_3.change_weight(&(source->C_3),eta);
	#ifdef	_DEBUG_T_
		std::cout<<"MP_2"<<std::endl;
	#endif
		MP_2.change_weight(&(source->MP_2),eta);
	#ifdef	_DEBUG_T_
		std::cout<<"C_1"<<std::endl;
	#endif
		C_1.change_weight(&(source->C_1),eta);
	}
	
	void copy_weight(Convolutional_Neural_Network *source)
	{
	#ifdef	_DEBUG_T_
		std::cout<<"FC_9"<<std::endl;
	#endif
		FC_9.copy_weight(&(source->FC_9));
	#ifdef	_DEBUG_T_
		std::cout<<"FC_8"<<std::endl;
	#endif
		FC_8.copy_weight(&(source->FC_8));
	#ifdef	_DEBUG_T_
		std::cout<<"FC_7"<<std::endl;
	#endif
		FC_7.copy_weight(&(source->FC_7));
	#ifdef	_DEBUG_T_
		std::cout<<"MP_6"<<std::endl;
	#endif
//		MP_6.copy_weight(&(source->MP_6));
	#ifdef	_DEBUG_T_
		std::cout<<"C_5"<<std::endl;
	#endif
//		C_5.copy_weight(&(source->C_5));
	#ifdef	_DEBUG_T_
		std::cout<<"MP_4"<<std::endl;
	#endif
		MP_4.copy_weight(&(source->MP_4));
	#ifdef	_DEBUG_T_
		std::cout<<"C_3"<<std::endl;
	#endif
		C_3.copy_weight(&(source->C_3));
	#ifdef	_DEBUG_T_
		std::cout<<"MP_2"<<std::endl;
	#endif
		MP_2.copy_weight(&(source->MP_2));
	#ifdef	_DEBUG_T_
		std::cout<<"C_1"<<std::endl;
	#endif
		C_1.copy_weight(&(source->C_1));
	}
};
