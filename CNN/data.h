/* 4 aixs */
class tube
{
  public:
    float ****d;
    int a_1, a_2, a_3, a_4;
    void init(int x, int y, int z, int a)
    {
	a_1 = x;
	a_2 = y;
	a_3 = z;
	a_4 = a;
	d = new float ***[x];
	for (int i = 0; i < x; i++)
	{
	    d[i] = new float **[y];
	    for (int j = 0; j < y; j++)
	    {
		d[i][j] = new float *[z];
		for (int k = 0; k < z; k++)
		{
		    d[i][j][k] = new float[a];
		    for (int l = 0; l < a; l++)
		    {
			d[i][j][k][l] = 0;
		    }
		}
	    }
	}
    }

    void init(int x, int y, int z, int a, float num)
    {
	a_1 = x;
	a_2 = y;
	a_3 = z;
	a_4 = a;
	d = new float ***[x];
	for (int i = 0; i < x; i++)
	{
	    d[i] = new float **[y];
	    for (int j = 0; j < y; j++)
	    {
		d[i][j] = new float *[z];
		for (int k = 0; k < z; k++)
		{
		    d[i][j][k] = new float[a];
		    for (int l = 0; l < a; l++)
		    {
			d[i][j][k][l] = R(num);
		    }
		}
	    }
	}
    }

    void add(tube *source)
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		for (int k = 0; k < a_3; k++)
		{
		    for (int l = 0; l < a_4; l++)
		    {
			d[i][j][k][l] += source->d[i][j][k][l];
		    }
		}
	    }
	}
    }

    void add(tube *source, float eta)
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		for (int k = 0; k < a_3; k++)
		{
		    for (int l = 0; l < a_4; l++)
		    {
			d[i][j][k][l] += eta * source->d[i][j][k][l];
		    }
		}
	    }
	}
    }

    void copy(tube *source)
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		for (int k = 0; k < a_3; k++)
		{
		    for (int l = 0; l < a_4; l++)
		    {
			d[i][j][k][l] = source->d[i][j][k][l];
		    }
		}
	    }
	}
    }

    void reset()
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		for (int k = 0; k < a_3; k++)
		{
		    for (int l = 0; l < a_4; l++)
		    {
			d[i][j][k][l] = 0;
		    }
		}
	    }
	}
    }
};

/* 3 aixs */
class cube
{
  public:
    float ***d;
    int a_1, a_2, a_3;

    void init(int x, int y, int z)
    {
	a_1 = x;
	a_2 = y;
	a_3 = z;
	d = new float **[x];
	for (int i = 0; i < x; i++)
	{
	    d[i] = new float *[y];
	    for (int j = 0; j < y; j++)
	    {
		d[i][j] = new float[z];
		for (int k = 0; k < z; k++)
		{
		    d[i][j][k] = 0;
		}
	    }
	}
    }

    void init(int x, int y, int z, float num)
    {
	a_1 = x;
	a_2 = y;
	a_3 = z;
	d = new float **[x];
	for (int i = 0; i < x; i++)
	{
	    d[i] = new float *[y];
	    for (int j = 0; j < y; j++)
	    {
		d[i][j] = new float[z];
		for (int k = 0; k < z; k++)
		{
		    d[i][j][k] = R(num);
		}
	    }
	}
    }

    void add(cube *source)
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		for (int k = 0; k < a_3; k++)
		{
		    d[i][j][k] += source->d[i][j][k];
		}
	    }
	}
    }

    void add(cube *source, float eta)
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		for (int k = 0; k < a_3; k++)
		{
		    d[i][j][k] += eta * source->d[i][j][k];
		}
	    }
	}
    }

    void copy(cube *source)
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		for (int k = 0; k < a_3; k++)
		{
		    d[i][j][k] = source->d[i][j][k];
		}
	    }
	}
    }

    void reset()
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		for (int k = 0; k < a_3; k++)
		{
		    d[i][j][k] = 0;
		}
	    }
	}
    }
};

/* 2 aixs */
class mat
{
  public:
    float **d;
    int a_1, a_2;

    void init(int x, int y)
    {
	a_1 = x;
	a_2 = y;
	d = new float *[x];
	for (int i = 0; i < x; i++)
	{
	    d[i] = new float[y];
	    for (int j = 0; j < y; j++)
	    {
		d[i][j] = 0;
	    }
	}
    }

    void init(int x, int y, float num)
    {
	a_1 = x;
	a_2 = y;
	d = new float *[x];
	for (int i = 0; i < x; i++)
	{
	    d[i] = new float[y];
	    for (int j = 0; j < y; j++)
	    {
		d[i][j] = R(num);
	    }
	}
    }

    void add(mat *source)
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		d[i][j] += source->d[i][j];
	    }
	}
    }

    void add(mat *source, float eta)
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		d[i][j] += eta * source->d[i][j];
	    }
	}
    }

    void copy(mat *source)
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		d[i][j] = source->d[i][j];
	    }
	}
    }

    void reset()
    {
	for (int i = 0; i < a_1; i++)
	{
	    for (int j = 0; j < a_2; j++)
	    {
		d[i][j] = 0;
	    }
	}
    }
};

class array
{
  public:
    float *d;
    int a_1;

    void init(int num)
    {
	a_1 = num;
	d = new float[num];
	for (int i = 0; i < num; i++)
	{
	    d[i] = 0;
	}
    }

    void init(int num, float n)
    {
	a_1 = num;
	d = new float[num];
	for (int i = 0; i < num; i++)
	{
	    d[i] = R(n);
	}
    }

    void add(array *source)
    {
	for (int i = 0; i < a_1; i++)
	{
	    d[i] += source->d[i];
	}
    }

    void add(array *source, float eta)
    {
	for (int i = 0; i < a_1; i++)
	{
	    d[i] += eta * source->d[i];
	}
    }

    void copy(array *source)
    {
	for (int i = 0; i < a_1; i++)
	{
	    d[i] = source->d[i];
	}
    }

    void reset()
    {
	for (int i = 0; i < a_1; i++)
	{
	    d[i] = 0;
	}
    }
};
