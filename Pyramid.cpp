#include <time.h>
#include <string>
#include "Pyramid.h"

#define ROTA 1
#define PI 3.14159
const double distant = sqrt(2.0) / 2;
const double length = sqrt(2.0);

//double Euler[3] = { (-PI/2) + 0, 0, PI/2 }; //pry
double coordinate[24][2] = {
	0.431336888576942,		0.0783548913495655,
	- 0.603865877072909,	0.164710743929735,
	- 0.147431397709338,	0.213292341515998,
	0.958776978777472,		0.225383995529468,
	0.636903910629166,		0.302095693044213,
	0.340107358201859,		0.317210918032203,
	- 0.399902318094066,	0.362954135521618,
	- 0.790749752895810,	0.370018820872982,
	0.0758148638725872,		0.383757972114521,
	- 0.164614545207328,	0.455605710451299,
	0.978101567416815,		0.467560727148095,
	0.499157082767137,		0.507994251215552,
	- 0.592755737517309,	0.517396052595896,
	0.741701615963150,		0.524036498904626,
	0.258507132744262,		0.547160679801918,
	- 0.360408975733460,	0.602611916086288,
	- 0.822579128799829,	0.610752739665504,
	0.0176391711036626,		0.619741832959460,
	0.919385889184784,		0.703987668445936,
	0.602118909851588,		0.732353169739320,
	- 0.586558596320368,	0.760062851739679,
	0.243590096118758,		0.789579806954302,
	- 0.188978613922738,	0.807812254673816,
	0.925083442354750,	0.946704291159700
};

double dodecahedron[20][2] = {
	0.500000000000000, 0.116139763599385,
	-0.500000000000000, 0.883860236400615,
	0.500000000000000, 0.883860236400615,
	-0.500000000000000, 0.116139763599385,
	0, 0.383860236400615,
	-1, 0.616139763599385,
	1, 0.383860236400615,
	0, 0.616139763599385,
	0.383860236400615, 0.500000000000000,
	-0.616139763599385, 0.500000000000000,
	-0.383860236400615, 0.500000000000000,
	0.616139763599385, 0.500000000000000,
	0.250000000000000, 0.304086723984696,
	-0.750000000000000, 0.695913276015304,
	0.250000000000000, 0.695913276015304,
	-0.750000000000000, 0.304086723984696,
	-0.250000000000000, 0.304086723984696,
	0.750000000000000, 0.695913276015304,
	-0.250000000000000, 0.695913276015304,
	0.750000000000000, 0.304086723984696,
};

double Euler[3] = { -PI*coordinate[23][1], 0, PI*coordinate[23][0] }; //pry
double m[3][3] = { cos(Euler[1])*cos(Euler[2]), -cos(Euler[1])*sin(Euler[2]), sin(Euler[1]),
cos(Euler[0])*sin(Euler[2]) + cos(Euler[2])*sin(Euler[0])*sin(Euler[1]), cos(Euler[0])*cos(Euler[2]) - sin(Euler[0])*sin(Euler[1])*sin(Euler[2]), -cos(Euler[1])*sin(Euler[0]),
sin(Euler[0])*sin(Euler[2]) - cos(Euler[0])*cos(Euler[2])*sin(Euler[1]), cos(Euler[2])*sin(Euler[0]) + cos(Euler[0])*sin(Euler[1])*sin(Euler[2]), cos(Euler[0])*cos(Euler[1])
};

//int Pyramid::tansform(double* v, unsigned char* eqdata, unsigned char* pyramidata, int w, int h)
int Pyramid::tansform(double* v, cv::Mat &Eqimg, cv::Mat& pyramidimg, int w, int h, int i, int j)
{
#if ROTA
	double u[3] = { v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
		v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
		v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2] };
	v[0] = u[0];
	v[1] = u[1];
	v[2] = u[2];
#endif
	double r = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

	double phi = abs(asin(v[2] / r) + PI / 2);
	double theta = abs(atan2(v[1], v[0]) + PI);  //atan2比atan更高级。

	double x = (phi / PI)*(h - 1);
	double y = (theta / (PI * 2)) * (w - 1);

	if (x > 0 && x < h - 1 && y > 0 && y < w - 1)
	{
		double u = abs(x - floor(x));
		double v = abs(y - floor(y));
		pyramidimg.at<cv::Vec3b>(i, width - 1 - j) = Eqimg.at<cv::Vec3b>(x, y)*(1 - u)*(1 - v)
			+ Eqimg.at<cv::Vec3b>(x, y + 1)*(1 - u)*(v)
			+ Eqimg.at<cv::Vec3b>(x + 1, y)*(u)*(1 - v)
			+ Eqimg.at<cv::Vec3b>(x + 1, y + 1)*(u)*(v);
	}
	else
	{
		pyramidimg.at<cv::Vec3b>(i, width - 1 - j) = Eqimg.at<cv::Vec3b>(x, y);//(pyramidheight - 1 - i, j)
	}

	return 0;
}

int Pyramid::Eq2Pyramid()
{
	char* filename = "eq.jpeg";
	cv::Mat Eq = cv::imread(filename);

	int w = Eq.size().width;
	int h = Eq.size().height;

	double alpha = atan(1 / distant) * 2;
	height = h*alpha/PI;
	width = h*alpha / PI;
	height /= 4;
	width /= 4;
	height *= 4;
	width *= 4;

	int unit = height / 2;
	double a = (double)height / width; //slope 


	time_t start = clock();
	cv::Mat pyramid = cv::Mat::zeros(cv::Size(height, width), CV_8UC3);
	for (int i = 0; i < height; ++i)
	{
		double py = (i - (height*0.5)) / unit;//行 y轴
		for (int j = 0; j < width; ++j)
		{
			double px = (j - (width*0.5)) / unit;//列 x轴

			if (py > a*px + 1)
			{
				double b = abs(py - a*px);
				double d = b*sqrt(0.5);
				double Vz = abs(sqrt(0.5) - d) / sqrt(0.5);//*length;//[0-1]  待改动
				double v[3] = { Vz + px, py - Vz, Vz*length - distant };
				tansform(v, Eq, pyramid, w, h, i, j);
				continue;
			}

			if (py < a*px - 1)
			{
				double b = abs(py - a*px);
				double d = b*sqrt(0.5);
				double Vz = abs(sqrt(0.5) - d) / sqrt(0.5);
				double v[3] = { px - Vz, py + Vz, Vz*length - distant };
				tansform(v, Eq, pyramid, w, h, i, j);
				continue;
			}

			if (py > -a*px + 1)
			{
				double b = abs(py + a*px);
				double d = b*sqrt(0.5);
				double Vz = abs(sqrt(0.5) - d) / sqrt(0.5);
				double v[3] = { px - Vz, py - Vz, Vz*length - distant };
				tansform(v, Eq, pyramid, w, h, i, j);

				continue;
			}

			if (py < -a*px - 1)
			{
				double b = abs(py + a*px);
				double d = b*sqrt(0.5);
				double Vz = abs(sqrt(0.5) - d) / sqrt(0.5);
				double v[3] = { px + Vz, py + Vz, Vz*length - distant };
				tansform(v, Eq, pyramid, w, h, i, j);

				continue;
			}

			double v[3] = { px, py, 0 - distant };
			tansform(v, Eq, pyramid, w, h, i, j);
		}
	}
	time_t end = clock();
	printf("the running time is : %f\n", double(end - start) / CLOCKS_PER_SEC);
	cv::imshow("pyramid", pyramid);
	cv::imwrite("pyramid.jpg", pyramid);
	cv::waitKey(0);
	return 0;
}