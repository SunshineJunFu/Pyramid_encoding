// TODO: Add OpenCL kernel code here.
#define PI 3.141592


inline void tansform(float *v, __global float *m, __global int* constdata, __global unsigned char* eqimgdata, __global unsigned char* pyimgdata)
{
	int i = constdata[0];
	int j = constdata[1];
	int k = constdata[9];
	
	int pywidth = constdata[2];
	int pyheight = constdata[3];
	int eqwidth = constdata[4];
	int eqheight = constdata[5];
	int eqwidthstep = constdata[6];
	int pywidthstep = constdata[7];
	
	int nchannels = constdata[8];
	
	float u[3] = { v[0] * m[9*k+0] + v[1] * m[9*k+3] + v[2] * m[9*k+6],
		v[0] * m[9*k+1] + v[1] * m[9*k+4] + v[2] * m[9*k+7],
		v[0] * m[9*k+2] + v[1] * m[9*k+5] + v[2] * m[9*k+8] };
	
	float r = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);

	float phi = fabs(asin(u[2] / r) + PI / 2);
	float theta = fabs(atan2(u[1], u[0]) + PI);  //atan2比atan更高级。

	float x = (phi / PI)*(eqheight - 1);
	float y = (theta / (PI * 2)) * (eqwidth - 1);

	if (x > 0 && x < eqheight - 1 && y > 0 && y < eqwidth - 1)
	{
		float u = fabs(x - floor(x));
		float v = fabs(y - floor(y));
		pyimgdata[k*pywidthstep*pyheight + i*pywidthstep + (pywidth - 1 - j)*nchannels + 0] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 0] * (1 - u)*(1 - v)
			+ eqimgdata[(int)x*eqwidthstep + (int)(y + 1)*nchannels + 0] * (1 - u)*(v)
			+ eqimgdata[(int)(x + 1)*eqwidthstep + (int)y*nchannels + 0] * (u)*(1 - v)
			+ eqimgdata[(int)(x + 1)*eqwidthstep + (int)(y + 1)*nchannels + 0] * (u)*(v);
		pyimgdata[k*pywidthstep*pyheight + i*pywidthstep + (pywidth - 1 - j)*nchannels + 1] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 1] * (1 - u)*(1 - v)
			+ eqimgdata[(int)x*eqwidthstep + (int)(y + 1)*nchannels + 1] * (1 - u)*(v)
			+ eqimgdata[(int)(x + 1)*eqwidthstep + (int)y*nchannels + 1] * (u)*(1 - v)
			+ eqimgdata[(int)(x + 1)*eqwidthstep + (int)(y + 1)*nchannels + 1] * (u)*(v);
		pyimgdata[k*pywidthstep*pyheight + i*pywidthstep + (pywidth - 1 - j)*nchannels + 2] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 2] * (1 - u)*(1 - v)
			+ eqimgdata[(int)x*eqwidthstep + (int)(y + 1)*nchannels + 2] * (1 - u)*(v)
			+ eqimgdata[(int)(x + 1)*eqwidthstep + (int)y*nchannels + 2] * (u)*(1 - v)
			+ eqimgdata[(int)(x + 1)*eqwidthstep + (int)(y + 1)*nchannels + 2] * (u)*(v);

	}
	else
	{

		pyimgdata[k*pywidthstep*pyheight + i*pywidthstep + (pywidth - 1 - j)*nchannels + 0] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 0];//(pyramidheight - 1 - i, j)
		pyimgdata[k*pywidthstep*pyheight + i*pywidthstep + (pywidth - 1 - j)*nchannels + 1] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 1];
		pyimgdata[k*pywidthstep*pyheight + i*pywidthstep + (pywidth - 1 - j)*nchannels + 2] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 2];
	}

}

#define vdistant (sqrt(2.0f) / 2)
#define tall sqrt(2.0f)

__kernel void pyramidproject20(__global unsigned char* eqimgdata, __global unsigned char* pyimgdata, __global int* constdata,  __global float* m)
{

	int pywidth = constdata[2];
	int pyheight = constdata[3];
	int eqwidth = constdata[4];
	int eqheight = constdata[5];

	
	int id = get_global_id(0);
	int k = id / (pywidth*pyheight);
	int i = (id % (pywidth*pyheight)) / pywidth;
	int j = (id % (pywidth*pyheight)) % pywidth;

	constdata[0] = i;
	constdata[1] = j;
	constdata[9] = k;

	float py = i*2.0f / pyheight - 1;//行 y轴
	float px = j*2.0f / pywidth - 1;//列 x轴

	if (py > px + 1)
	{
	
		float b = fabs(py - px);

		float Vz = fabs(1-b);
		float v[3] = { Vz + px, py - Vz, Vz*tall - vdistant };

		tansform(v, m, constdata, eqimgdata, pyimgdata);
		
	}
	else
	{
		if (py < px - 1)
		{
			float b = fabs(py - px);
	
			float Vz = fabs(1-b);
			float v[3] = { px - Vz, py + Vz, Vz*tall - vdistant };
	
			tansform(v, m, constdata, eqimgdata, pyimgdata);
		}
		else
		{
			if (py > -px + 1)
			{
				float b = fabs(py + px);

				float Vz = fabs(1-b);
				float v[3] = { px - Vz, py - Vz, Vz*tall - vdistant };
	
				tansform(v, m, constdata, eqimgdata, pyimgdata);

			}
			else
			{
				if (py < -px - 1)
				{
					float b = fabs(py + px);
	
					float Vz = fabs(1-b);
					float v[3] = { px + Vz, py + Vz, Vz*tall - vdistant };
		
					tansform(v, m, constdata, eqimgdata, pyimgdata);

				}
				else
				{
					float v[3] = { px, py, -vdistant };
					tansform(v, m, constdata, eqimgdata, pyimgdata);
				}
			}
		}
	}
}

__kernel void pyramidproject(__global unsigned char* eqimgdata, __global unsigned char* pyimgdata, __global int* constdata,  __global float* m)
{

	int pywidth = constdata[2];
	int pyheight = constdata[3];
	int eqwidth = constdata[4];
	int eqheight = constdata[5];

//	const float vdistant = sqrt(2.0f) / 2;//view point
//	const float tall  = sqrt(2.0f);//
//	float a = 1.0f;

	//float unit = (float)(pyheight*0.5);

	int id = get_global_id(0);
	int i = id/pywidth;
	int j = id%pywidth;

	constdata[0] = i;
	constdata[1] = j;

	//float py = (i - (pyheight*0.5)) / unit;//行 y轴
	//float px = (j - (pywidth*0.5)) / unit;//列 x轴
	float py = i*2.0f / pyheight - 1;//行 y轴
	float px = j*2.0f / pywidth - 1;//列 x轴

	if (py > px + 1)
	{
	
		float b = fabs(py - px);
		//float d = b*sqrt(0.5);
		//float Vz = fabs(sqrt(0.5) - d) / sqrt(0.5);//*length;//[0-1]  待改动
		float Vz = fabs(1-b);
		float v[3] = { Vz + px, py - Vz, Vz*tall - vdistant };

		tansform(v, m, constdata, eqimgdata, pyimgdata);

		
	}
	else
	{
		if (py < px - 1)
		{
			float b = fabs(py - px);
			//float d = b*sqrt(0.5);
			//float Vz = fabs(sqrt(0.5) - d) / sqrt(0.5);
			float Vz = fabs(1-b);
			float v[3] = { px - Vz, py + Vz, Vz*tall - vdistant };
	
			tansform(v, m, constdata, eqimgdata, pyimgdata);
		}
		else
		{
			if (py > -px + 1)
			{
				float b = fabs(py + px);
				//float d = b*sqrt(0.5);
				//float Vz = fabs(sqrt(0.5) - d) / sqrt(0.5);
				float Vz = fabs(1-b);
				float v[3] = { px - Vz, py - Vz, Vz*tall - vdistant };
	
				tansform(v, m, constdata, eqimgdata, pyimgdata);

			}
			else
			{
				if (py < -px - 1)
				{
					float b = fabs(py + px);
					//float d = b*sqrt(0.5);
					//float Vz = fabs(sqrt(0.5) - d) / sqrt(0.5);
					float Vz = fabs(1-b);
					float v[3] = { px + Vz, py + Vz, Vz*tall - vdistant };
		
					tansform(v, m, constdata, eqimgdata, pyimgdata);

				}
				else
				{
					float v[3] = { px, py, -vdistant };
					tansform(v, m, constdata, eqimgdata, pyimgdata);
				}
			}
		}
	}
}
