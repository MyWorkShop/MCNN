class tube;
class cube;
class mat;
class array;
class Input_Layer;
class Convolutional_Layer;
class Max_Pooling_Layer;
class Fully_Connected_Layer;
class Output_Layer;

/* Activation Function */

namespace tan_h
{
float y(float x)
{
    float a = exp(x);
    float b = 1 / a;
    return ((a - b) / (a + b));
}

float df(float y)
{
    return (1 - y * y);
}
}

namespace softplus
{
float y(float x)
{
    return (log(exp(x) + 1));
}

float df(float y)
{
    return (1 - exp(-y));
}
}

/* layer class */

class Basic_Layer
{
  public:
    cube y;
    int num;
    int m, n;
};

class Input_Layer : public Basic_Layer
{
  public:
    void init(int num_pic, int a, int b)
    {
	num = num_pic;
	m = a;
	n = b;
	y.init(num, m, n);
    }
};

class Calculate_Layer : public Basic_Layer
{
  public:
    Basic_Layer *last_layer;
    int a, b;
    int next_m, next_n;
    int next_a, next_b;
    int next_num;
    int last_num;
    virtual void calculate_y();
    virtual void calculate_delta();
};

class Convolutional_Layer : public Calculate_Layer
{
  public:
    tube w;
    tube d_w;
    int a, b;
    cube delta;
    array bias;
    array d_bias;
    cube connect;
    int use;
    int last_m, last_n;
    int last_num;
    Max_Pooling_Layer *next_layer;
    bool s;
    float dropout;
    bool dropouted;

    float core(int x_aixs, int y_aixs, int num_source, int num_output);

    void init_1(Basic_Layer *last, int num_pics, int a_core, int b_core, bool start, int u, float dp);

    void init_2(Max_Pooling_Layer *next);

    void calculate_y();

    void calculate_delta();

    void calculate_d_w();

    void change_weight(float eta);

    void change_weight(Convolutional_Layer *source, float eta);

    void copy_weight(Convolutional_Layer *source);

    void dout();

    void udout();
};

class Max_Pooling_Layer : public Calculate_Layer
{
  public:
    array beta, d_beta;
    array bias, d_bias;
    int a, b;
    void *next_layer;
    bool Fully_Connected;
    cube d;
    cube delta;

    float max(int no, int x_aixs, int y_aixs);

    void init_1(Basic_Layer *last, int a_core, int b_core);

    void init_2(void *next, bool fcl);

    void calculate_y();

    void calculate_delta();

    void calculate_d_w();

    void change_weight(float eta);

    void change_weight(Max_Pooling_Layer *source, float eta);

    void copy_weight(Max_Pooling_Layer *source);
};

class Fully_Connected_Layer
{
  public:
    void *w;
    void *d_w;
    array bias;
    array d_bias;
    int last_m, last_n;
    int last_num;
    int next_num;
    int num;
    float *y;
    void *last_layer;
    void *next_layer;
    bool max_pooling;
    bool end;
    bool out;
    array delta;
    array connect;
    float dropout;
    bool dropouted;

    void init_1(void *last, int n, bool mpl, float dp)
    {
	num = n;
	last_layer = last;
	max_pooling = mpl;
	y = new float[num];
	dropout = dp;
	if (max_pooling == true)
	{
	    w = new cube;
	    d_w = new cube;
	    last_m = ((Calculate_Layer *)last_layer)->m;
	    last_n = ((Calculate_Layer *)last_layer)->n;
	    last_num = ((Calculate_Layer *)last_layer)->num;
	    ((tube *)w)->init(last_num, last_m, last_n, n, 1.0);
	    ((tube *)d_w)->init(last_num, last_m, last_n, n);
	}
	else
	{
	    w = new mat;
	    d_w = new mat;
	    last_num = ((Fully_Connected_Layer *)last_layer)->num;
	    ((mat *)w)->init(((Fully_Connected_Layer *)last_layer)->num, n, 1.0);
	    ((mat *)d_w)->init(((Fully_Connected_Layer *)last_layer)->num, n);
	}
	delta.init(num);
	bias.init(num, 1.0);
	d_bias.init(num);
	connect.init(num);
#ifdef _DEBUG_INIT_
	std::cout << "--\nFully_Connected_Layer Init Stage 1:" << std::endl;
	std::cout << "n,last_num,max_pooling" << std::endl;
	std::cout << n << ',' << last_num << ',' << max_pooling << std::endl;
#endif
    }

    void init_2(void *next, bool o);

    void calculate_y()
    {
	if (dropouted == true)
	{
	    if (max_pooling == true)
	    {
		for (int i = 0; i < num; i++)
		{
		    if (connect.d[i] == 1)
		    {
			float sum = 0;
			for (int j = 0; j < last_num; j++)
			{
			    for (int k = 0; k < last_m; k++)
			    {
				for (int l = 0; l < last_n; l++)
				{
				    sum += ((tube *)w)->d[j][k][l][i] * ((Max_Pooling_Layer *)last_layer)->y.d[j][k][l];
				}
			    }
			}
			y[i] = tan_h::y(sum + bias.d[i]);
		    }
		    else
		    {
			y[i] = -1;
		    }
		}
	    }
	    else
	    {
		for (int i = 0; i < num; i++)
		{
		    if (connect.d[i] == 1)
		    {
			float sum = 0;
			for (int j = 0; j < last_num; j++)
			{
			    sum += ((mat *)w)->d[j][i] * ((Fully_Connected_Layer *)last_layer)->y[j];
			}
			y[i] = tan_h::y(sum + bias.d[i]);
		    }
		    else
		    {
			y[i] = -1;
		    }
		}
	    }
	}
	else
	{
	    if (max_pooling == true)
	    {
		for (int i = 0; i < num; i++)
		{
		    float sum = 0;
		    for (int j = 0; j < last_num; j++)
		    {
			for (int k = 0; k < last_m; k++)
			{
			    for (int l = 0; l < last_n; l++)
			    {
				sum += ((tube *)w)->d[j][k][l][i] * ((Max_Pooling_Layer *)last_layer)->y.d[j][k][l];
			    }
			}
		    }
		    y[i] = tan_h::y(dropout * sum + bias.d[i]);
		}
	    }
	    else
	    {
		for (int i = 0; i < num; i++)
		{
		    float sum = 0;
		    for (int j = 0; j < last_num; j++)
		    {
			sum += ((mat *)w)->d[j][i] * ((Fully_Connected_Layer *)last_layer)->y[j];
		    }
		    y[i] = tan_h::y(dropout * sum + bias.d[i]);
		}
	    }
	}
    }

    void calculate_delta(float *d);

    void calculate_d_w()
    {
	if (max_pooling == false)
	{
	    for (int i = 0; i < num; i++)
	    {
		for (int j = 0; j < last_num; j++)
		{
		    ((mat *)d_w)->d[j][i] += delta.d[i] * ((Fully_Connected_Layer *)last_layer)->y[j];
		}

		d_bias.d[i] += delta.d[i];
	    }
	}
	else
	{
	    for (int i = 0; i < num; i++)
	    {
		for (int j = 0; j < last_num; j++)
		{
		    for (int k = 0; k < last_m; k++)
		    {
			for (int l = 0; l < last_n; l++)
			{
			    ((tube *)d_w)->d[j][k][l][i] += delta.d[i] * ((Max_Pooling_Layer *)last_layer)->y.d[j][k][l];
			}
		    }
		}

		d_bias.d[i] += delta.d[i];
	    }
	}
    }

    void change_weight(float eta)
    {
	if (max_pooling == false)
	{
	    ((mat *)w)->add(((mat *)d_w), eta / dropout);
	    bias.add(&d_bias, eta / dropout);
	    ((mat *)d_w)->reset();
	    d_bias.reset();
	}
	else
	{
	    ((tube *)w)->add(((tube *)d_w), eta / dropout);
	    bias.add(&d_bias, eta / dropout);
	    ((tube *)d_w)->reset();
	    d_bias.reset();
	}
    }

    void change_weight(Fully_Connected_Layer *source, float eta)
    {
	if (max_pooling == false)
	{
	    ((mat *)w)->add(((mat *)source->d_w), eta / dropout);
	    bias.add(&(source->d_bias), eta / dropout);
	    ((mat *)source->d_w)->reset();
	    source->d_bias.reset();
	}
	else
	{
	    ((tube *)w)->add(((tube *)source->d_w), eta / dropout);
	    bias.add(&(source->d_bias), eta / dropout);
	    ((tube *)source->d_w)->reset();
	    source->d_bias.reset();
	}
    }

    void copy_weight(Fully_Connected_Layer *source)
    {
	if (max_pooling == false)
	{
	    ((mat *)w)->copy(((mat *)source->w));
	    bias.copy(&(source->bias));
	}
	else
	{
	    ((tube *)w)->copy(((tube *)source->w));
	    bias.copy(&(source->bias));
	}
    }

    void dout()
    {
	dropouted = true;
	for (int i = 0; i < num; i++)
	{
	    if (r() <= dropout)
	    {
		connect.d[i] = 1;
	    }
	    else
	    {
		connect.d[i] = 0;
	    }
	}
    }

    void udout()
    {
	dropouted = false;
	for (int i = 0; i < num; i++)
	{
	    connect.d[i] = 1;
	}
    }
};

class Output_Layer
{
  public:
    /* w */
    mat w;
    mat d_w;
    /* b */
    array b;
    array d_b;
    /* last */
    int last_m, last_n;
    int last_num;
    Fully_Connected_Layer *last_layer;
    /* this */
    int num;
    float *y;
    array delta;

    void init_1(void *last, int n)
    {
	num = n;
	last_layer = (Fully_Connected_Layer *)last;
	y = new float[num];
	last_num = last_layer->num;
	w.init(last_layer->num, n, 1.0);
	d_w.init(last_layer->num, n);
	b.init(n, 3.0);
	d_b.init(n);
	delta.init(num);
#ifdef _DEBUG_INIT_
	std::cout << "--\nOutput_Layer Init:" << std::endl;
	std::cout << "n,last_num" << std::endl;
	std::cout << n << ',' << last_num << ',' << std::endl;
#endif
    }

    void calculate_y()
    {
	float sum_w = 0;
	for (int i = 0; i < num; i++)
	{
	    y[i] = 0;
	    for (int j = 0; j < last_num; j++)
	    {
		y[i] += w.d[j][i] * last_layer->y[j];
	    }
	    sum_w += exp(y[i] + b.d[i]);
	}
	for (int i = 0; i < num; i++)
	{
	    y[i] = exp(y[i] + b.d[i]) / sum_w;
	}
    }

    void calculate_delta(float *d)
    {
	for (int i = 0; i < num; i++)
	{
	    delta.d[i] = (d[i] - y[i]);
	}
    }

    void calculate_d_w()
    {
	for (int i = 0; i < num; i++)
	{
	    for (int j = 0; j < last_num; j++)
	    {
		d_w.d[j][i] += delta.d[i] * ((Fully_Connected_Layer *)last_layer)->y[j];
	    }
	    d_b.d[i] += delta.d[i];
	}
    }

    void change_weight(float eta)
    {
	w.add(&d_w, eta);
	b.add(&d_b, eta);
	d_w.reset();
	d_b.reset();
    }

    void change_weight(Output_Layer *source, float eta)
    {
	w.add(&(source->d_w), eta);
	b.add(&(source->d_b), eta);
	source->d_w.reset();
	source->d_b.reset();
    }

    void copy_weight(Output_Layer *source)
    {
	w.copy(&(source->w));
	b.copy(&(source->b));
    }
};

/* layer function */

void Convolutional_Layer::calculate_y()
{
    if (dropouted == true)
    {
	for (int i = 0; i < num; i++)
	{
	    for (int j = 0; j < m; j++)
	    {
		for (int k = 0; k < n; k++)
		{
		    if (connect.d[i][j][k] == 1)
		    {
			float sum = 0;
			for (int l = 0; l < last_num; l++)
			{
			    sum += core(j, k, l, i);
			}
			y.d[i][j][k] = softplus::y(sum + bias.d[i]);
		    }
		    else
		    {
			y.d[i][j][k] = 0;
		    }
		}
	    }
	}
    }
    else
    {
	for (int i = 0; i < num; i++)
	{
	    for (int j = 0; j < m; j++)
	    {
		for (int k = 0; k < n; k++)
		{
		    float sum = 0;
		    for (int l = 0; l < last_num; l++)
		    {
			sum += core(j, k, l, i);
		    }
		    y.d[i][j][k] = softplus::y(dropout * sum + bias.d[i]);
		}
	    }
	}
    }
}

void Convolutional_Layer::dout()
{
    dropouted = true;
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		if (r() <= dropout)
		{
		    connect.d[i][j][k] = 1;
		}
		else
		{
		    connect.d[i][j][k] = 0;
		}
	    }
	}
    }
}

void Convolutional_Layer::udout()
{
    dropouted = false;
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		connect.d[i][j][k] = 1;
	    }
	}
    }
}

void Fully_Connected_Layer::init_2(void *next, bool o)
{
    if (next == NULL)
    {
	next_layer = NULL;
	end = true;
    }
    else
    {
	next_layer = next;
	end = false;
	out = o;
	if (o == false)
	{
	    next_num = ((Fully_Connected_Layer *)next_layer)->num;
	}
	else
	{
	    next_num = ((Output_Layer *)next_layer)->num;
	}
    }
#ifdef _DEBUG_INIT_
    std::cout << "--\nFully_Connected_Layer Init Stage 2:" << std::endl;
    std::cout << "end,next_num,out" << std::endl;
    std::cout << end << ',' << next_num << ',' << out << std::endl;
#endif
}

void Fully_Connected_Layer::calculate_delta(float *d)
{
    if (end == true)
    {
	for (int i = 0; i < num; i++)
	{
	    delta.d[i] = tan_h::df(y[i]) * (d[i] - y[i]);
	}
    }
    else if (out == false)
    {
	for (int i = 0; i < num; i++)
	{
	    float sum = 0;
	    for (int j = 0; j < next_num; j++)
	    {
		sum += ((mat *)((Fully_Connected_Layer *)next_layer)->w)->d[i][j] * ((Fully_Connected_Layer *)next_layer)->delta.d[j];
	    }
	    delta.d[i] = sum * tan_h::df(y[i]);
	}
    }
    else
    {
	for (int i = 0; i < num; i++)
	{
	    float sum = 0;
	    for (int j = 0; j < next_num; j++)
	    {
		sum += (((Output_Layer *)next_layer)->w).d[i][j] * ((Output_Layer *)next_layer)->delta.d[j];
	    }
	    delta.d[i] = sum * tan_h::df(y[i]);
	}
    }
}

void Convolutional_Layer::init_1(Basic_Layer *last, int num_pics, int a_core, int b_core, bool start, int u, float dp)
{
    s = start;
    num = num_pics;
    a = a_core;
    b = b_core;
    last_layer = last;
    use = u;
    dropout = dp;
    dropouted = false;
    last_num = last_layer->num;
    last_n = last_layer->n;
    last_m = last_layer->m;
    m = last_m - a + 1;
    n = last_n - a + 1;
    w.init(num, last_num, a, b, 2.0);
    bias.init(num, 3.0);
    d_w.init(num, last_num, a, b);
    d_bias.init(num);
    y.init(num, m, n);
    delta.init(num, m, n);
    connect.init(num, m, n);
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		connect.d[i][j][k] = 1;
	    }
	}
    }
#ifdef _DEBUG_INIT_
    std::cout << "--\nConvolutional_Layer Init Stage 1:" << std::endl;
    std::cout << "s,num,use,a,b,m,n" << std::endl;
    std::cout << s << ',' << num << ',' << use << ',' << a << ',' << b << ',' << m << ',' << n << std::endl;
#endif
}

void Convolutional_Layer::init_2(Max_Pooling_Layer *next)
{
    next_layer = next;
    next_num = next->num;
    next_a = next->a;
    next_b = next->b;
#ifdef _DEBUG_INIT_
    std::cout << "--\nConvolutional_Layer Init Stage 2:" << std::endl;
    std::cout << "next_num,next_a,next_b" << std::endl;
    std::cout << next_num << ',' << next_a << ',' << next_b << std::endl;
#endif
}

float Convolutional_Layer::core(int x_aixs, int y_aixs, int num_source, int num_output)
{
    float sum = 0;
    for (int i = 0; i < a; i++)
    {
	for (int j = 0; j < b; j++)
	{
	    sum += w.d[num_output][num_source][i][j] * last_layer->y.d[num_source][x_aixs + i][y_aixs + j];
	}
    }
    return (sum);
}

void Max_Pooling_Layer::init_1(Basic_Layer *last, int a_core, int b_core)
{
    num = last->num;
    a = a_core;
    b = b_core;
    m = last->m / a;
    n = last->n / b;
    last_layer = last;
    beta.init(num, 1.0);
    bias.init(num, 3.0);
    d_beta.init(num);
    d_bias.init(num);
    y.init(num, m, n);
    d.init(num, m, n);
    delta.init(num, m, n);
#ifdef _DEBUG_INIT_
    std::cout << "--\nMax_Pooling_Layer Init Stage 1:" << std::endl;
    std::cout << "num,m,n" << std::endl;
    std::cout << num << ',' << m << ',' << n << std::endl;
#endif
}

void Max_Pooling_Layer::init_2(void *next, bool fcl)
{
    Fully_Connected = fcl;
    next_layer = next;
    if (Fully_Connected == true)
    {
	next_num = ((Fully_Connected_Layer *)next_layer)->num;
    }
    else
    {
	next_num = ((Convolutional_Layer *)next_layer)->num;
	next_m = ((Convolutional_Layer *)next_layer)->m;
	next_n = ((Convolutional_Layer *)next_layer)->n;
	next_a = ((Convolutional_Layer *)next_layer)->a;
	next_b = ((Convolutional_Layer *)next_layer)->b;
    }
#ifdef _DEBUG_INIT_
    std::cout << "--\nMax_Pooling_Layer Init Stage 2:" << std::endl;
    std::cout << "Fully_Connected" << std::endl;
    std::cout << Fully_Connected << std::endl;
#endif
}

float Max_Pooling_Layer::max(int no, int x_aixs, int y_aixs)
{
    float max_float = last_layer->y.d[no][x_aixs * a][y_aixs * b];
    for (int i = 0; i < a; i++)
    {
	for (int j = 0; j < b; j++)
	{
	    if (last_layer->y.d[no][x_aixs * a + i][y_aixs * b + j] >= max_float)
	    {
		max_float = last_layer->y.d[no][x_aixs * a + i][y_aixs * b + j];
	    }
	}
    }
    return (max_float);
}

void Max_Pooling_Layer::calculate_y()
{
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		d.d[i][j][k] = max(i, j, k);
		y.d[i][j][k] = tan_h::y(beta.d[i] * d.d[i][j][k] + bias.d[i]);
	    }
	}
    }
}

void Max_Pooling_Layer::calculate_delta()
{
    if (Fully_Connected == true)
    {
	for (int i = 0; i < num; i++)
	{
	    for (int j = 0; j < m; j++)
	    {
		for (int k = 0; k < n; k++)
		{
		    float sum = 0;
		    for (int l = 0; l < next_num; l++)
		    {
			sum += ((Fully_Connected_Layer *)next_layer)->delta.d[l] * ((tube *)((Fully_Connected_Layer *)next_layer)->w)->d[i][j][k][l];
		    }
		    delta.d[i][j][k] = sum * tan_h::df(y.d[i][j][k]);
		}
	    }
	}
    }
    else
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
	    for (int j = 0; j < ((Convolutional_Layer *)next_layer)->m; j++)
	    {
		for (int k = 0; k < ((Convolutional_Layer *)next_layer)->n; k++)
		{
		    for (int l_1 = 0; l_1 < next_num; l_1++)
		    {
			for (int l_2 = 0; l_2 < ((Convolutional_Layer *)next_layer)->a; l_2++)
			{
			    for (int l_3 = 0; l_3 < ((Convolutional_Layer *)next_layer)->b; l_3++)
			    {
				delta.d[i][j + l_2][k + l_3] += ((Convolutional_Layer *)next_layer)->delta.d[l_1][j][k] * ((Convolutional_Layer *)next_layer)->w.d[l_1][i][l_2][l_3];
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
		}
	    }
	}
    }
}

void Max_Pooling_Layer::calculate_d_w()
{
    for (int i = 0; i < num; i++)
    {
	float sum = 0;
	float sum_ = 0;
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		sum += delta.d[i][j][k] * d.d[i][j][k];
		sum_ += delta.d[i][j][k];
	    }
	}

	d_beta.d[i] += sum;

	d_bias.d[i] += sum_;
    }
}

void Max_Pooling_Layer::change_weight(float eta)
{
    beta.add(&d_beta, eta);
    bias.add(&d_bias, eta);
    d_beta.reset();
    d_bias.reset();
}

void Max_Pooling_Layer::change_weight(Max_Pooling_Layer *source, float eta)
{
    beta.add(&(source->d_beta), eta);
    bias.add(&(source->d_bias), eta);
    source->d_beta.reset();
    source->d_bias.reset();
}

void Max_Pooling_Layer::copy_weight(Max_Pooling_Layer *source)
{
    beta.copy(&(source->beta));
    bias.copy(&(source->bias));
}

void Convolutional_Layer::calculate_delta()
{
    for (int i = 0; i < num; i++)
    {
	for (int j = 0; j < m; j++)
	{
	    for (int k = 0; k < n; k++)
	    {
		delta.d[i][j][k] = next_layer->beta.d[i] * next_layer->delta.d[i][j / next_a][k / next_b];
	    }
	    for (int k = 0; k < n; k++)
	    {
		delta.d[i][j][k] = softplus::df(y.d[i][j][k]) * delta.d[i][j][k];
	    }
	}
    }
}

void Convolutional_Layer::calculate_d_w()
{
    if (s == true)
    {
	for (int i = 0; i < num; i++)
	{
	    for (int j = 0; j < a; j++)
	    {
		for (int k = 0; k < b; k++)
		{
		    for (int l_1 = 0; l_1 < last_num; l_1++)
		    {
			float sum = 0;
			for (int l_2 = 0; l_2 < m; l_2++)
			{
			    for (int l_3 = 0; l_3 < n; l_3++)
			    {
				sum += ((Input_Layer *)last_layer)->y.d[l_1][l_2 + j][l_3 + k] * delta.d[i][l_2][l_3];
			    }
			}

			d_w.d[i][l_1][j][k] += sum;
		    }
		}
	    }
	    float sum = 0;
	    for (int j = 0; j < m; j++)
	    {
		for (int k = 0; k < n; k++)
		{
		    sum += delta.d[i][j][k];
		}
	    }

	    d_bias.d[i] += sum;
	}
    }
    else
    {
	for (int i = 0; i < num; i++)
	{
	    for (int j = 0; j < a; j++)
	    {
		for (int k = 0; k < b; k++)
		{
		    for (int l_1 = 0; l_1 < last_num; l_1++)
		    {
			float sum = 0;
			for (int l_2 = 0; l_2 < m; l_2++)
			{
			    for (int l_3 = 0; l_3 < n; l_3++)
			    {
				sum += ((Max_Pooling_Layer *)last_layer)->y.d[l_1][l_2 + j][l_3 + k] * delta.d[i][l_2][l_3];
			    }
			}

			d_w.d[i][l_1][j][k] += sum;
		    }
		}
	    }
	    float sum = 0;
	    for (int j = 0; j < m; j++)
	    {
		for (int k = 0; k < n; k++)
		{
		    sum += delta.d[i][j][k];
		}
	    }

	    d_bias.d[i] += sum;
	}
    }
}

void Convolutional_Layer::change_weight(float eta)
{
    w.add(&d_w, eta / dropout);
    bias.add(&d_bias, eta / dropout);
    d_w.reset();
    d_bias.reset();
}

void Convolutional_Layer::change_weight(Convolutional_Layer *source, float eta)
{
    w.add(&(source->d_w), eta / dropout);
    bias.add(&(source->d_bias), eta / dropout);
    (source->d_w).reset();
    (source->d_bias).reset();
}

void Convolutional_Layer::copy_weight(Convolutional_Layer *source)
{
    w.copy(&(source->w));
    bias.copy(&(source->bias));
}
