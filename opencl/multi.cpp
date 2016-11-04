#include <time.h>
#include <stdio.h>	
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>	
#endif
#define PERSPECTIVE_NUM 20
#define MAX_SOURCE_SIZE (0x100000)	
#define PI 3.14159f

enum IndexName
{
	IDi = 0,
	IDj,
	IDpywidth,
	IDpyheight,
	IDeqwidth,
	IDeqheight,
	IDeqwidthstep,
	IDpywidthstep,
	IDnchannels,
	IDk
};

void initialization(const int* constdata);
int pyramid_opencl(int* constdata, char* eqdata, char* pydata[]);
void release();

const float vdistant = sqrt(2.0f) / 2;//view point
const float tall = sqrt(2.0f);//

static cl_platform_id platform_id = NULL;
static cl_device_id device_id = NULL;
static cl_context context = NULL;
static cl_command_queue command_queue = NULL;

static cl_program program = NULL;
static cl_kernel kernel = NULL;
static cl_uint ret_num_devices;
static cl_uint ret_num_platforms;
static cl_int ret;
static size_t source_size;
static char *source_str;

/*Create Buffer Object */
static cl_mem rotamatrixmobj = NULL;
static cl_mem constdatamobj = NULL;
static cl_mem eqdatamobj = NULL;
static cl_mem pydatamobj = NULL;

static size_t global_item_size = 1;
const size_t local_item_size = 1;

void initialization(const int* constdata)
{
	FILE *fp;
	const char fileName[] = "E:\\ES_ALG_PyramidProjection\\trunk\\Pyramid_projection_opencl\\pyramidproject.cl";

	/* Load kernel source file */
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	/* Get Platform/Device Information*/
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	/* Create OpenCL Context */
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	/*Create command queue */
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);


	/* Create rota matrix*/
	double dodecahedron[PERSPECTIVE_NUM][2] = { 0.500000000000000, 0.116139763599385, -0.500000000000000, 0.883860236400615, 0.500000000000000, 0.883860236400615,
		-0.500000000000000, 0.116139763599385,		0, 0.383860236400615,		-1, 0.616139763599385,		1, 0.383860236400615,		0, 0.616139763599385,
		0.383860236400615, 0.500000000000000,		-0.616139763599385, 0.500000000000000,		-0.383860236400615, 0.500000000000000,
		0.616139763599385, 0.500000000000000,		0.250000000000000, 0.304086723984696,		-0.750000000000000, 0.695913276015304,
		0.250000000000000, 0.695913276015304,		-0.750000000000000, 0.304086723984696,		-0.250000000000000, 0.304086723984696,
		0.750000000000000, 0.695913276015304,		-0.250000000000000, 0.695913276015304,		0.750000000000000, 0.304086723984696	};

	float rotamatrix20[PERSPECTIVE_NUM*9] = { 0 };
	for (int i = 0; i < PERSPECTIVE_NUM; ++i)
	{
		float Euler[3] = { -PI*dodecahedron[i][1], 0.0f, PI * dodecahedron[i][0] }; //pry
		float rotamatrix[9] = { cos(Euler[1])*cos(Euler[2]), -cos(Euler[1])*sin(Euler[2]), sin(Euler[1]),
			cos(Euler[0])*sin(Euler[2]) + cos(Euler[2])*sin(Euler[0])*sin(Euler[1]), cos(Euler[0])*cos(Euler[2]) - sin(Euler[0])*sin(Euler[1])*sin(Euler[2]), -cos(Euler[1])*sin(Euler[0]),
			sin(Euler[0])*sin(Euler[2]) - cos(Euler[0])*cos(Euler[2])*sin(Euler[1]), cos(Euler[2])*sin(Euler[0]) + cos(Euler[0])*sin(Euler[1])*sin(Euler[2]), cos(Euler[0])*cos(Euler[1])
		};
		for (int j = 0; j < 9; ++j)
		{
			rotamatrix20[9 * i + j] = rotamatrix[j];
		}
	}

	/*Create Buffer Object */
	rotamatrixmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 180 * sizeof(float), NULL, &ret);
	constdatamobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 10 * sizeof(int), NULL, &ret);
	eqdatamobj = clCreateBuffer(context, CL_MEM_READ_WRITE, constdata[IDeqwidth] * constdata[IDeqheight] * constdata[IDnchannels] * sizeof(unsigned char), NULL, &ret);
	pydatamobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 20 * constdata[IDpywidth] * constdata[IDpyheight] * constdata[IDnchannels] * sizeof(unsigned char), NULL, &ret);


	/* Copy input data to the memory buffer */
	ret = clEnqueueWriteBuffer(command_queue, rotamatrixmobj, CL_TRUE, 0, PERSPECTIVE_NUM * 9 * sizeof(float), rotamatrix20, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, constdatamobj, CL_TRUE, 0, 10 * sizeof(int), constdata, 0, NULL, NULL);

	/* Create kernel program from source file*/
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	/* Create data parallel OpenCL kernel */
	kernel = clCreateKernel(program, "pyramidproject20", &ret);

	/* Set OpenCL kernel arguments */
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&pydatamobj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&constdatamobj);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&rotamatrixmobj);
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&eqdatamobj);

	global_item_size = constdata[IDpyheight] * constdata[IDpywidth] * PERSPECTIVE_NUM;

	return;
}


void release()
{
	/* Finalization */
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(constdatamobj);
	ret = clReleaseMemObject(eqdatamobj);
	ret = clReleaseMemObject(pydatamobj);
	ret = clReleaseMemObject(rotamatrixmobj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(source_str);
	return;
}


int pyramid_opencl(int* constdata, char* eqdata, char* pydata[])
{
	/* Write eq buffer */
	ret = clEnqueueWriteBuffer(command_queue, eqdatamobj, CL_TRUE, 0, constdata[IDeqwidth] * constdata[IDeqheight] * constdata[IDnchannels] * sizeof(unsigned char), eqdata, 0, NULL, NULL);
	
	time_t start = clock();

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		&global_item_size, &local_item_size, 0, NULL, NULL);
	
	/* Transfer result to host */
	for (int i = 0; i < PERSPECTIVE_NUM; ++i)
	{
		ret = clEnqueueReadBuffer(command_queue, pydatamobj, CL_TRUE, constdata[IDpywidth] * constdata[IDpyheight] * constdata[IDnchannels] * sizeof(unsigned char)*i,
			constdata[IDpywidth] * constdata[IDpyheight] * constdata[IDnchannels] * sizeof(unsigned char), pydata[i], 0, NULL, NULL);
	}
	time_t end = clock();
	printf("the running time is : %f\n", double(end - start) / CLOCKS_PER_SEC);



	return 0;


}

int main()
{
	/* Initialize input data */
	IplImage *eqimg = cvLoadImage("eq.jpeg", CV_LOAD_IMAGE_UNCHANGED);
	int eqwidth = eqimg->width;
	int eqheight = eqimg->height;
	int eqnchannels = eqimg->nChannels;
	int eqwidthstep = eqimg->widthStep;

	float alpha = atan(1.f / vdistant) * 2;
	int pyheight = (int)(eqheight*alpha / PI);
	int pywidth = (int)(eqheight*alpha / PI);
	pyheight /= 4;
	pywidth /= 4;
	pyheight *= 4;
	pywidth *= 4;

	//创建20幅图
	char* pyimg_data[PERSPECTIVE_NUM];
	IplImage* pyramidimg[PERSPECTIVE_NUM];
	
	for (int i = 0; i < PERSPECTIVE_NUM; ++i)
	{
		pyramidimg[i] = cvCreateImage(cvSize(pywidth, pyheight), eqimg->depth, eqimg->nChannels);
		pyimg_data[i] = pyramidimg[i]->imageData;

	}
	int *constdata;
	constdata = (int *)malloc(10 * sizeof(int));
	constdata[0] = constdata[1] = constdata[9] = 0;
	constdata[2] = pywidth;
	constdata[3] = pyheight;
	constdata[4] = eqwidth;
	constdata[5] = eqheight;
	constdata[6] = eqwidthstep;
	constdata[7] = pyramidimg[0]->widthStep;
	constdata[8] = eqnchannels;
	//函数接口
	// 初始化
	initialization(constdata);

	/* eq layout transform into pyramid format
	输入：constdata是一个常量数组，包括图像高宽，通道数
		eq->imgdata是eq图像数据首地址
	输出：pyimg_data是一个大小为20的数值指针，指向输出的20幅pyramid*/
	//while (1)
	{
		pyramid_opencl(constdata, eqimg->imageData, pyimg_data);
	}
	//display and save pyimg_data result
	for (int i = 0; i < PERSPECTIVE_NUM; ++i)
	{
		std::string fn = "pyramid" + std::to_string(i) + ".jpg";
		cvShowImage("pyramid", pyramidimg[i]);
		cvSaveImage(fn.c_str(), pyramidimg[i]);
		cvWaitKey(0);
	}

	release();

	free(constdata);
	return 0;
}