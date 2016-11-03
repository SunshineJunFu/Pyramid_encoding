
#if 1
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

#define MAX_SOURCE_SIZE (0x100000)	
#define PI 3.14159f
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
	double dodecahedron[20][2] = {0.500000000000000, 0.116139763599385,		-0.500000000000000, 0.883860236400615,		0.500000000000000, 0.883860236400615,
		-0.500000000000000, 0.116139763599385,		0, 0.383860236400615,		-1, 0.616139763599385,		1, 0.383860236400615,		0, 0.616139763599385,
		0.383860236400615, 0.500000000000000,		-0.616139763599385, 0.500000000000000,		-0.383860236400615, 0.500000000000000,
		0.616139763599385, 0.500000000000000,		0.250000000000000, 0.304086723984696,		-0.750000000000000, 0.695913276015304,
		0.250000000000000, 0.695913276015304,		-0.750000000000000, 0.304086723984696,		-0.250000000000000, 0.304086723984696,
		0.750000000000000, 0.695913276015304,		-0.250000000000000, 0.695913276015304,		0.750000000000000, 0.304086723984696	};

	float rotamatrix20[180] = { 0 };
	for (int i = 0; i < 20; ++i)
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
	ret = clEnqueueWriteBuffer(command_queue, rotamatrixmobj, CL_TRUE, 0, 180 * sizeof(float), rotamatrix20, 0, NULL, NULL);
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

	global_item_size = constdata[IDpyheight]*constdata[IDpywidth] * 20;

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

void display(int w, int h, int depth, int nchannels)
{
	/* display result */
	//IplImage *eqimg = cvLoadImage("eq.jpeg", CV_LOAD_IMAGE_UNCHANGED);
	IplImage* pyramidimg = cvCreateImage(cvSize(w, h), depth, nchannels);
	std::string fn;
	for (int i = 0; i < 20; ++i)
	{
		fn = "pyramid" + std::to_string(i) + ".jpg";
		ret = clEnqueueReadBuffer(command_queue, pydatamobj, CL_TRUE,
			w * h * nchannels * sizeof(unsigned char) * i,
			w * h * nchannels * sizeof(unsigned char), pyramidimg->imageData, 0, NULL, NULL);
		cvShowImage("pyramid", pyramidimg);
		cvSaveImage(fn.c_str(), pyramidimg);
		cvWaitKey(0);
	}
	return;
}

int pyramid_opencl(int* constdata, char* eqdata, char* pydata)
{
	/* Write eq buffer */
	ret = clEnqueueWriteBuffer(command_queue, eqdatamobj, CL_TRUE, 0, constdata[IDeqwidth] * constdata[IDeqheight] * constdata[IDnchannels] * sizeof(unsigned char), eqdata, 0, NULL, NULL);
	
	time_t start = clock();

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		&global_item_size, &local_item_size, 0, NULL, NULL);
	//ret = clEnqueueReadBuffer(command_queue, constdatamobj, CL_TRUE, 0, 10 * sizeof(int), constdata, 0, NULL, NULL);
	
	/* Transfer result to host */
	ret = clEnqueueReadBuffer(command_queue, pydatamobj, CL_TRUE,0,
		constdata[IDpywidth] * constdata[IDpyheight] * constdata[IDnchannels] * sizeof(unsigned char), pydata, 0, NULL, NULL);
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

	int unit = pyheight / 2;
	float a = (float)pyheight / pywidth; //slope 
	IplImage* pyramidimg = cvCreateImage(cvSize(pywidth, pyheight), eqimg->depth, eqimg->nChannels);

	int *constdata;
	constdata = (int *)malloc(10 * sizeof(int));
	constdata[0] = constdata[1] = constdata[9] = 0;
	constdata[2] = pywidth;
	constdata[3] = pyheight;
	constdata[4] = eqwidth;
	constdata[5] = eqheight;
	constdata[6] = eqwidthstep;
	constdata[7] = pyramidimg->widthStep;
	constdata[8] = eqnchannels;
	//º¯Êý½Ó¿Ú

	initialization(constdata);

	pyramid_opencl(constdata, eqimg->imageData, pyramidimg->imageData);

	display(pyramidimg->width, pyramidimg->height, pyramidimg->depth, pyramidimg->nChannels);

	release();

	free(constdata);
	//cvShowImage("pyramid", pyramidimg);
	//cvSaveImage("pyramid.jpg", pyramidimg);
	//cvWaitKey(0);
	
	return 0;
}

#endif