/*#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#define input_f "pic.jpg"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
   Mat org_img = imread(input_f, 1);
   Mat src1(org_img.rows, org_img.cols, CV_8UC3);
   Mat src2(org_img.rows, org_img.cols, CV_8UC3);
   Mat dst(org_img.rows, org_img.cols, CV_8UC3);
   for (int i = 0; i < 50; i = i + 2 ) {
	  int i2 = 2*i;
	  int i2_1 = 23;// *i + 1;
	  blur(org_img, src1, Size(i2, i2), Point(-1, -1), 4);
	  blur(org_img, src2, Size(i2_1, i2_1), Point(-1, -1), 4);

	  addWeighted(src1, -100, src2, 100, 0.0, dst);
	  imshow("Low freq _img ", dst);
	  waitKey(6000);
   }
   return 0;
}
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

int main(int argc, char** argv)
{

	Mat src = imread("pic1.jpg", 1);
	Mat src1(src.rows, src.cols, CV_8UC3);
	Mat src2(src.rows, src.cols, CV_8UC3);
	Mat src3(src.rows, src.cols, CV_8UC3);
	Mat dst1, dst2 ;
	Mat dst1_1, dst1_2;
	char zBuffer[35];

	/*
	for (int i = 1; i < 5000; i = i + 5)
	{
		int i1 = i;
		int i1_1 = i+5;// 2 * i + 1;// *i + 1;
		int i1_2 = i + 2;// 2 * i + 1;// *i + 1;
		blur(src, src1, Size(i1, i1), Point(-1, -1), 4);
		blur(src, src2, Size(i1_1, i1_1), Point(-1, -1), 4);
		blur(src, src3, Size(i1_2, i1_2), Point(-1, -1), 4);
		addWeighted(src1, -100, src2, 100, 0.0, dst1_1);
		addWeighted(src2, -100, src3, 100, 0.0, dst1_2);
		addWeighted(dst1_1, 1, dst1_2,  1, 0, dst1);
		addWeighted(dst1_1, 1, src, 0.4, 0, dst2);
		imshow("Low freq _img ", dst2);


		//wait for 2 seconds
		int c = waitKey(100);

		//if the "esc" key is pressed during the wait, return
		if (c == 27)
		{
			return 0;
		}
	}
	*/
	blur(src, src1, Size(1, 1), Point(-1, -1), 4);
	blur(src, src2, Size(2, 2), Point(-1, -1), 4);
	blur(src, src3, Size(3, 3), Point(-1, -1), 4);
	addWeighted(src1, -1, src2, 1, 0.0, dst1_1);
	addWeighted(src2, -1, src3, 1, 0.0, dst1_2);
	addWeighted(dst1_1, +5, dst1_2, 5, 0.0, dst1);
	imwrite("1_diff.jpg", dst1_1); 
	imwrite("2_diff.jpg", dst1_2);
	imwrite("sum_diff.jpg", dst1);
	return 0;

}
