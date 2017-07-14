#define SCNN_SIZE 8

class CSC_Layer : public Calculate_Layer
{
  public:
    Convolutional_Layer *next_layer;
    Small_Convolutional_Neural_Network ***fliter;
    cube delta;

    void init_1(Basic_Layer *last, int num_pics, int a_core, int b_core);

    void init_2(Convolutional_Layer *next);

    void calculate_y();

    void calculate_delta();

    void calculate_d_w();

    void change_weight(float eta);

    void change_weight(CSC_Layer *source, float eta);

    void copy_weight(CSC_Layer *source);
};

void CSC_Layer::init_1(Basic_Layer *last, int num_pics, int a_core, int b_core)
{
    num = num_pics;
    last_layer = last;
    last_num = last_layer->num;
    a = a_core;
    b = b_core;
    m = (last_layer->m - SCNN_SIZE) / a + 1;
    n = (last_layer->n - SCNN_SIZE) / b + 1;
    y.init(num, m, n);
    delta.init(num, m, n);
    fliter = new Small_Convolutional_Neural_Network **[num];
    for (int i = 0; i < num; i++)
    {
	fliter[i] = new Small_Convolutional_Neural_Network *[m];
	for (int j = 0; j < m; j++)
	{
	    fliter[i][j] = new Small_Convolutional_Neural_Network[n];
	    for (int k = 0; k < n; k++)
	    {
		fliter[i][j][k].init(last_num);
	    }
	}
    }
#ifdef _DEBUG_INIT_
    std::cout
	<< "--\nCSC_Layer Init Stage 1:" << std::endl;
    std::cout << "last_num,num_pics,a_core,b_core,m,n" << std::endl;
    std::cout << last_num << ',' << num_pics << ',' << a_core << ',' << b_core << ',' << m << ',' << n << std::endl;
#endif
}

void CSC_Layer::init_2(Convolutional_Layer *next)
{
    next_layer = next;
    next_num = next_layer->num;
}

void CSC_Layer::calculate_y()
{
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		for (int l_1 = 0; l_1 < SCNN_SIZE; l_1++)
		{
		    for (int l_2 = 0; l_2 < SCNN_SIZE; l_2++)
		    {
			for (int l_3 = 0; l_3 < last_num; l_3++)
			{
			    fliter[i][j][k].INPUT.y.d[l_3][l_1][l_2] = last_layer->y.d[l_3][l_1 + j * a][l_2 + k * b];
			}
		    }
		}
		fliter[i][j][k].calculate();
		y.d[i][j][k] = fliter[i][j][k].FC_3.y[0];
	    }
	}
    }
}

void CSC_Layer::calculate_delta()
{
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		delta.d[i][j][k] = 0;
	    }
	}
    }
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < next_layer->m; j++)
	{
	    for (int k = 0; k < next_layer->n; k++)
	    {
		for (int l_1 = 0; l_1 < next_num; l_1++)
		{
		    for (int l_2 = 0; l_2 < next_layer->a; l_2++)
		    {
			for (int l_3 = 0; l_3 < next_layer->b; l_3++)
			{
			    delta.d[i][j + l_2][k + l_3] += next_layer->delta.d[l_1][j][k] * next_layer->w.d[l_1][i][l_2][l_3];
			}
		    }
		}
	    }
	}
    }
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		delta.d[i][j][k] = delta.d[i][j][k] * tan_h::df(y.d[i][j][k]);
		fliter[i][j][k].FC_3.delta.d[0] = delta.d[i][j][k];
		fliter[i][j][k].calculate_delta();
	    }
	}
    }
}

void CSC_Layer::calculate_d_w()
{
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		fliter[i][j][k].calculate_d_w();
	    }
	}
    }
}

void CSC_Layer::change_weight(float eta)
{
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		fliter[i][0][0].change_weight(&fliter[i][j][k],eta);
	    }
	}
    }
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		fliter[i][j][k].copy_weight(&fliter[i][0][0]);
	    }
	}
    }
}

void CSC_Layer::change_weight(CSC_Layer *source, float eta)
{
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		fliter[i][0][0].change_weight(&(source->fliter[i][j][k]), eta);
	    }
	}
    }
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		fliter[i][j][k].copy_weight(&fliter[i][0][0]);
	    }
	}
    }
}

void CSC_Layer::copy_weight(CSC_Layer *source)
{
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		fliter[i][j][k].copy_weight(&(source->fliter[i][j][k]));
	    }
	}
    }
}
