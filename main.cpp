#include <opencv2\opencv.hpp>
#include <string>
#define ROTA 0
int sign(double x)
{
	if (x == 0)
	{
		return 1;
	}
	else
	{
		if (x > 0) return 1;
		else return -1;
	}
}
int main()
{
	double pi = 3.14159;
	double Euler[3] = { -pi/2, 0, 0 / 2 }; //pyr

	double m[3][3] = { cos(Euler[1])*cos(Euler[2]), -cos(Euler[1])*sin(Euler[2]), sin(Euler[1]),
		cos(Euler[0])*sin(Euler[2]) + cos(Euler[2])*sin(Euler[0])*sin(Euler[1]), cos(Euler[0])*cos(Euler[2]) - sin(Euler[0])*sin(Euler[1])*sin(Euler[2]), -cos(Euler[1])*sin(Euler[0]),
		sin(Euler[0])*sin(Euler[2]) - cos(Euler[0])*cos(Euler[2])*sin(Euler[1]), cos(Euler[2])*sin(Euler[0]) + cos(Euler[0])*sin(Euler[1])*sin(Euler[2]), cos(Euler[0])*cos(Euler[1])
	};


	/*std::cout << atan(pi / -4) << std::endl;
	std::cout << atan2(pi,-4) << std::endl;
	system("pause");
	return 0;*/
	std::string filename = "eq.jpeg";
	cv::Mat Eq = cv::imread(filename);
	int pyramidheight = 1000;
	int pyramidwidth = 1000;
	cv::Mat pyramid = cv::Mat::zeros(cv::Size(pyramidheight, pyramidwidth), CV_8UC3);
	int w = Eq.size().width;
	int h = Eq.size().height;
	//std::cout << __LINE__ << std::endl;
	double distant = sqrt(2.0)/2;
	double length = 2;// sqrt(2.0);
	for (int i = 0; i < pyramidheight; ++i)
	{
		double py = i - pyramidheight/2;

		for (int j = 0; j < pyramidwidth; ++j)
		{
			double px = j - pyramidwidth / 2;

			if (py > px + pyramidheight/2)
			{
				double b = (py - px) / (pyramidwidth / 2);
				double bx = -b / 2;
				double by = b / 2;
				double d = sqrt(bx*bx + by*by);
				double Vz = abs(sqrt(2) / 2 - d) / (sqrt(2) / 2) *length;

				double v[3] = { (abs(Vz / length) + (px / (pyramidwidth / 2)))*abs((Vz / length) - 1), 
					((py / (pyramidheight/2)) - abs(Vz / length))*abs((Vz / length) - 1), 
					Vz - distant };
#if ROTA
				double u[3] = { v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0], v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1], v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2] };
				v[0] = u[0];
				v[1] = u[1];
				v[2] = u[2];
#endif
				double r = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

				double phi = asin(v[2] / r) + pi / 2;
				double theta = atan2(v[1], v[0]) + pi;// sign(abs(v[0]))*pi;  //atan2比atang更高级。

				int x = (phi / pi)*h - 0.5;
				int y = (theta / (pi * 2)) * w - 0.5 + w / 2;;
				if (y > w)
					y -= w;

				if (x < 0)
					x = 0;
				if (y < 0)
					y = 0;
				if (x > h - 1)
					x = h - 1;
				if (y > w - 1)
					y = w - 1;
				//if (Vz  < 0.4)
				{
				//	std::cout << "phi " << phi << " theta" << theta << std::endl;
				//	std::cout << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
				////	std::cout << "x " << x << " y" << y << std::endl;
				}
				
				pyramid.at<cv::Vec3b>(pyramidheight - 1 - i, j) = Eq.at<cv::Vec3b>(x, y);
				continue;
			}
			if (py < px - pyramidheight/2)
			{
				double b = (py - px) / (pyramidwidth / 2);
				double bx = b / 2;
				double by = -b / 2;
				double d = sqrt(bx*bx + by*by);
				double Vz = abs(sqrt(2) / 2 - d) / (sqrt(2) / 2) *length;

				double v[3] = { ((px / (pyramidwidth / 2)) - abs(Vz / length))*abs((Vz / length) - 1), 
					((py / (pyramidheight / 2)) + abs(Vz / length))*abs((Vz / length) - 1), 
					Vz - distant };
			/*	double v[3] = { (abs(Vz / length) + (px / (pyramidwidth / 2)))*abs((Vz / length) - 1), 
					((py / (pyramidheight/2)) - abs(Vz / length))*abs((Vz / length) - 1), 
					Vz - distant };*/

#if ROTA
				double u[3] = { v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0], v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1], v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2] };
				v[0] = u[0];
				v[1] = u[1];
				v[2] = u[2];
#endif
				double r = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

				double phi = asin(v[2] / r) + pi / 2;
				double theta = atan2(v[1], v[0]) + pi;// sign(abs(v[0]))*pi;  //atan2比atang更高级。

				int x = (phi / pi)*h - 0.5;
				int y = (theta / (pi * 2)) * w - 0.5 + w / 2;;
				if (y > w)
					y -= w;

				if (x < 0)
					x = 0;
				if (y < 0)
					y = 0;
				if (x > h - 1)
					x = h - 1;
				if (y > w - 1)
					y = w - 1;
				//if (Vz  < 0.4)
				{
					//	std::cout << "phi " << phi << " theta" << theta << std::endl;
				//	std::cout << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
					////	std::cout << "x " << x << " y" << y << std::endl;
				}

				pyramid.at<cv::Vec3b>(pyramidheight - 1 - i, j) = Eq.at<cv::Vec3b>(x, y);
				continue;
			}
			if (py > -px + pyramidheight/2)
			{
				double b = (py + px) / (pyramidwidth / 2);
				double bx = -b / 2;
				double by = b / 2;
				double d = sqrt(bx*bx + by*by);
				double Vz = abs(sqrt(2) / 2 - d) / (sqrt(2) / 2) *length;

				double v[3] = { ((px / (pyramidwidth / 2)) - abs(Vz / length))*abs((Vz / length) - 1),
					((py / (pyramidheight / 2)) - abs(Vz / length))*abs((Vz / length) - 1), 
					Vz - distant };
#if ROTA
				double u[3] = { v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0], v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1], v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2] };
				v[0] = u[0];
				v[1] = u[1];
				v[2] = u[2];
#endif
				double r = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

				double phi = asin(v[2] / r) + pi / 2;
				double theta = atan2(v[1], v[0]) + pi;// sign(abs(v[0]))*pi;  //atan2比atang更高级。

				int x = (phi / pi)*h - 0.5;
				int y = (theta / (pi * 2)) * w - 0.5 + w / 2;;
				if (y > w)
					y -= w;

				if (x < 0)
					x = 0;
				if (y < 0)
					y = 0;
				if (x > h - 1)
					x = h - 1;
				if (y > w - 1)
					y = w - 1;
				//if (Vz  < 0.4)
				{
					//	std::cout << "phi " << phi << " theta" << theta << std::endl;
				//	std::cout << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
					////	std::cout << "x " << x << " y" << y << std::endl;
				}

				pyramid.at<cv::Vec3b>(pyramidheight - 1 - i, j) = Eq.at<cv::Vec3b>(x, y);
				continue;
			}
			if (py < -px - pyramidheight/2)
			{
				double b = (py + px) / (pyramidwidth / 2);
				double bx = b / 2;
				double by = -b / 2;
				double d = sqrt(bx*bx + by*by);
				double Vz = abs(sqrt(2) / 2 - d) / (sqrt(2) / 2) *length;

				double v[3] = { ((px / (pyramidwidth / 2)) + abs(Vz / length))*abs((Vz / length) - 1), 
					((py / (pyramidheight / 2)) + abs(Vz / length))*abs((Vz / length) - 1), 
					Vz - distant };
#if ROTA
				double u[3] = { v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0], v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1], v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2] };
				v[0] = u[0];
				v[1] = u[1];
				v[2] = u[2];
#endif
				double r = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

				double phi = asin(v[2] / r) + pi / 2;
				double theta = atan2(v[1], v[0]) + pi;// sign(abs(v[0]))*pi;  //atan2比atang更高级。

				int x = (phi / pi)*h - 0.5;
				int y = (theta / (pi * 2)) * w - 0.5 + w / 2;;
				if (y > w)
					y -= w;

				if (x < 0)
					x = 0;
				if (y < 0)
					y = 0;
				if (x > h - 1)
					x = h - 1;
				if (y > w - 1)
					y = w - 1;
				//if (Vz  < 0.4)
				{
					//	std::cout << "phi " << phi << " theta" << theta << std::endl;
				//	std::cout << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
					////	std::cout << "x " << x << " y" << y << std::endl;
				}

				pyramid.at<cv::Vec3b>(pyramidheight - 1 - i, j) = Eq.at<cv::Vec3b>(x, y);
				continue;
			}

			double v[3] = { (px) / (pyramidheight / 2.0), (py) / (pyramidheight/2.0), 0 - distant };
			double r = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
#if ROTA
			double u[3] = { v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0], v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1], v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2] };
			v[0] = u[0];
			v[1] = u[1];
			v[2] = u[2];
#endif
			double phi = asin(v[2] / r) + pi / 2;
			double theta = atan2(v[1], v[0]) + pi;// sign(abs(v[0]))*pi;  //atan2比atang更高级。
			
			int x = (phi / pi)*h - 0.5;
			int y = (theta / (pi * 2)) * w - 0.5 + w / 2;;
			if (y > w)
				y -= w;

			if (x < 0)
				x = 0;
			if (y < 0)
				y = 0;
			if (x > h -1)
				x = h -1;
			if (y > w -1)
				y = w -1;

			//if (j == 512)
			//{
			//	std::cout << "phi " << phi << " theta" << theta << std::endl;
			////	std::cout << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;
			////	std::cout << "x " << x << " y" << y << std::endl;
			//}
			pyramid.at<cv::Vec3b>(pyramidheight -1 - i, j) = Eq.at<cv::Vec3b>(x, y);

		}
	}
	cv::imshow("pyramid", pyramid);
	cv::waitKey(0);
	cv::imwrite("pyramid.jpg", pyramid);
	return 0;
}