#ifndef PYRAMID_H_
#define PYRAMID_H_

#include <opencv2\opencv.hpp>

class Pyramid{
public:
	double* rotamatrix;
	int Eq2Pyramid();
private:
	int height;
	int width;
	//int tansform(double* v, unsigned char* eqdata, unsigned char* pyramidata, int w, int h);
	int tansform(double* v, cv::Mat &Eqimg, cv::Mat& pyramidimg, int w, int h, int i, int j);
};


#endif