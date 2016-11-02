
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



int pyramid_opencl(int* constdata, char* eqdata, char* pydata)
{
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;

	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;


	FILE *fp;
	const char fileName[] = "E:\\ES_ALG_PyramidProjection\\trunk\\Pyramid_projection_opencl\\pyramidproject.cl";
	size_t source_size;
	char *source_str;

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


	
	float Euler[3] = { -PI*0.5, 0.0f, PI * 0.5 }; //pry
	float rotamatrix[9] = { cos(Euler[1])*cos(Euler[2]), -cos(Euler[1])*sin(Euler[2]), sin(Euler[1]),
		cos(Euler[0])*sin(Euler[2]) + cos(Euler[2])*sin(Euler[0])*sin(Euler[1]), cos(Euler[0])*cos(Euler[2]) - sin(Euler[0])*sin(Euler[1])*sin(Euler[2]), -cos(Euler[1])*sin(Euler[0]),
		sin(Euler[0])*sin(Euler[2]) - cos(Euler[0])*cos(Euler[2])*sin(Euler[1]), cos(Euler[2])*sin(Euler[0]) + cos(Euler[0])*sin(Euler[1])*sin(Euler[2]), cos(Euler[0])*cos(Euler[1])
	};
	
	int pywidth = constdata[2];
	int pyheight = constdata[3];
	int eqwidth = constdata[4];
	int eqheight = constdata[5];
	int eqnchannels = constdata[8];

	/*Create Buffer Object */
	cl_mem rotamatrixmobj = NULL;
	cl_mem constdatamobj = NULL;
	cl_mem eqdatamobj = NULL;
	cl_mem pydatamobj = NULL;

	rotamatrixmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 9 * sizeof(float), NULL, &ret);
	constdatamobj = clCreateBuffer(context, CL_MEM_READ_WRITE, 9 * sizeof(int), NULL, &ret);
	eqdatamobj = clCreateBuffer(context, CL_MEM_READ_WRITE, eqwidth * eqheight * eqnchannels * sizeof(unsigned char), NULL, &ret);
	pydatamobj = clCreateBuffer(context, CL_MEM_READ_WRITE, pywidth * pyheight * eqnchannels * sizeof(unsigned char), NULL, &ret);

	/* Copy input data to the memory buffer */
	ret = clEnqueueWriteBuffer(command_queue, rotamatrixmobj, CL_TRUE, 0, 9 * sizeof(float), rotamatrix, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, constdatamobj, CL_TRUE, 0, 9 * sizeof(int), constdata, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, eqdatamobj, CL_TRUE, 0, eqwidth * eqheight * eqnchannels * sizeof(unsigned char), eqdata, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, pydatamobj, CL_TRUE, 0, pywidth * pyheight * eqnchannels * sizeof(unsigned char), eqdata, 0, NULL, NULL);

	/* Create kernel program from source file*/
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	/* Create data parallel OpenCL kernel */
	kernel = clCreateKernel(program, "pyramidproject", &ret);

	/* Set OpenCL kernel arguments */
	
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&eqdatamobj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&pydatamobj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&constdatamobj);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&rotamatrixmobj);

	size_t global_item_size = pyheight*pywidth;
	size_t local_item_size = 1;
	time_t start = clock();

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		&global_item_size, &local_item_size, 0, NULL, NULL);

	/* Transfer result to host */
	ret = clEnqueueReadBuffer(command_queue, pydatamobj, CL_TRUE, 0, pywidth * pyheight * eqnchannels * sizeof(unsigned char), pydata, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, constdatamobj, CL_TRUE, 0, 9 * sizeof(int), constdata, 0, NULL, NULL);
	time_t end = clock();
	printf("the running time is : %f\n", double(end - start) / CLOCKS_PER_SEC);
	/* Display Results */
	
	for (int i = 0; i < 9; i++) {
		printf("%d ", constdata[i]);

	}
	
	/* Finalization */
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(constdatamobj);
	ret = clReleaseMemObject(eqdatamobj);
	ret = clReleaseMemObject(pydatamobj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	free(source_str);
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
	constdata = (int *)malloc(9 * sizeof(int));
	constdata[0] = constdata[1] = 0;
	constdata[2] = pywidth;
	constdata[3] = pyheight;
	constdata[4] = eqwidth;
	constdata[5] = eqheight;
	constdata[6] = eqwidthstep;
	constdata[7] = pyramidimg->widthStep;
	constdata[8] = eqnchannels;
	//º¯Êý½Ó¿Ú
	pyramid_opencl(constdata, eqimg->imageData, pyramidimg->imageData);
	free(constdata);
	cvShowImage("pyramid", pyramidimg);
	cvSaveImage("pyramid.jpg", pyramidimg);
	cvWaitKey(0);
	return 0;
}

#endif