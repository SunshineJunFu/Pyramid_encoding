// TODO: Add OpenCL kernel code here.

void tansform(float *v, __global float *m, __global int* constdata, __global unsigned char* eqimgdata, __global unsigned char* pyimgdata)
{

	float PI = 3.14159;
	int i = constdata[0];
	int j = constdata[1];
	int pywidth = constdata[2];
	int pyheight = constdata[3];
	int eqwidth = constdata[4];
	int eqheight = constdata[5];
	int eqwidthstep = constdata[6];
	int pywidthstep = constdata[7];
	int nchannels = constdata[8];
	
	float u[3] = { v[0] * m[0] + v[1] * m[3] + v[2] * m[6],
		v[0] * m[1] + v[1] * m[4] + v[2] * m[7],
		v[0] * m[2] + v[1] * m[5] + v[2] * m[8] };
	
	float r = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);

	float phi = fabs(asin(u[2] / r) + PI / 2);
	float theta = fabs(atan2(u[1], u[0]) + PI);  //atan2比atan更高级。

	float x = (phi / PI)*(eqheight - 1);
	float y = (theta / (PI * 2)) * (eqwidth - 1);

	if (x > 0 && x < eqheight - 1 && y > 0 && y < eqwidth - 1)
	{
		//printf("789\n");
		float u = fabs(x - floor(x));
		float v = fabs(y - floor(y));
		pyimgdata[i*pywidthstep + (pywidth - 1 - j)*nchannels + 0] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 0] * (1 - u)*(1 - v)
			+ eqimgdata[(int)x*eqwidthstep + (int)(y + 1)*nchannels + 0] * (1 - u)*(v)
			+eqimgdata[(int)(x + 1)*eqwidthstep + (int)y*nchannels + 0] * (u)*(1 - v)
			+ eqimgdata[(int)(x + 1)*eqwidthstep + (int)(y + 1)*nchannels + 0] * (u)*(v);
		pyimgdata[i*pywidthstep + (pywidth - 1 - j)*nchannels + 1] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 1] * (1 - u)*(1 - v)
			+ eqimgdata[(int)x*eqwidthstep + (int)(y + 1)*nchannels + 1] * (1 - u)*(v)
			+eqimgdata[(int)(x + 1)*eqwidthstep + (int)y*nchannels + 1] * (u)*(1 - v)
			+ eqimgdata[(int)(x + 1)*eqwidthstep + (int)(y + 1)*nchannels + 1] * (u)*(v);
		pyimgdata[i*pywidthstep + (pywidth - 1 - j)*nchannels + 2] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 2] * (1 - u)*(1 - v)
			+ eqimgdata[(int)x*eqwidthstep + (int)(y + 1)*nchannels + 2] * (1 - u)*(v)
			+eqimgdata[(int)(x + 1)*eqwidthstep + (int)y*nchannels + 2] * (u)*(1 - v)
			+ eqimgdata[(int)(x + 1)*eqwidthstep + (int)(y + 1)*nchannels + 2] * (u)*(v);

	}
	else
	{

		pyimgdata[i*pywidthstep + (pywidth - 1 - j)*nchannels + 0] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 0];//(pyramidheight - 1 - i, j)
		pyimgdata[i*pywidthstep + (pywidth - 1 - j)*nchannels + 1] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 1];
		pyimgdata[i*pywidthstep + (pywidth - 1 - j)*nchannels + 2] = eqimgdata[(int)x*eqwidthstep + (int)y*nchannels + 2];
	}

}



__kernel void pyramidproject(__global unsigned char* eqimgdata, __global unsigned char* pyimgdata, __global int* constdata,  __global float* m)
{

	//printf("123\n");


	int pywidth = constdata[2];
	int pyheight = constdata[3];
	int eqwidth = constdata[4];
	int eqheight = constdata[5];

	float a = 1.0f;
	float vdistant = sqrt(2.0f) / 2;//view point
	float tall = sqrt(2.0f);//
	float unit = (float)(pyheight*0.5);

	int id = get_global_id(0);
	int i = id/pywidth;
	int j = id%pywidth;

	constdata[0] = i;
	constdata[1] = j;

	float py = (i - (pyheight*0.5)) / unit;//行 y轴
	float px = (j - (pywidth*0.5)) / unit;//列 x轴

	if (py > a*px + 1)
	{
	
		float b = fabs(py - a*px);
		float d = b*sqrt(0.5);
		float Vz = fabs(sqrt(0.5) - d) / sqrt(0.5);//*length;//[0-1]  待改动
		float v[3] = { Vz + px, py - Vz, Vz*tall - vdistant };

		tansform(v, m, constdata, eqimgdata, pyimgdata);

		
	}
	else
	{
		if (py < a*px - 1)
		{
			float b = fabs(py - a*px);
			float d = b*sqrt(0.5);
			float Vz = fabs(sqrt(0.5) - d) / sqrt(0.5);
			float v[3] = { px - Vz, py + Vz, Vz*tall - vdistant };
	
			tansform(v, m, constdata, eqimgdata, pyimgdata);
		}
		else
		{
			if (py > -a*px + 1)
			{
				float b = fabs(py + a*px);
				float d = b*sqrt(0.5);
				float Vz = fabs(sqrt(0.5) - d) / sqrt(0.5);
				float v[3] = { px - Vz, py - Vz, Vz*tall - vdistant };
	
				tansform(v, m, constdata, eqimgdata, pyimgdata);

			}
			else
			{
				if (py < -a*px - 1)
				{
					float b = fabs(py + a*px);
					float d = b*sqrt(0.5);
					float Vz = fabs(sqrt(0.5) - d) / sqrt(0.5);
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

