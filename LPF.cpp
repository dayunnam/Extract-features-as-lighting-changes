#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#define input_f "pic2.jpg"
#define output_f "대역통과_이미지.jpg"

using namespace cv;

int main(int argc, char** argv)
{
	namedWindow("Before", CV_WINDOW_AUTOSIZE);


	// Load the source image
	Mat org_img = imread(input_f, 1);
	Mat inverse_img;
	Mat LQ_img1;
	Mat LQ_img2;
	Mat LQ_img3;
	Mat LQ_img4;
	Mat LQ_img5;
	Mat LQ_img6;
	Mat LQ_img7;
	Mat LQ_img8;





	Mat LQ_inverse_img;
	Mat sum_img;
	Mat LQ_sum_img;
	Mat LQ_sum_img1;
	Mat LQ_sum_img2;
	Mat LQ_sum_inverse_img;
	Mat HQ_inverse_img;
	int sigma = 2;

	IplImage *org_ = new IplImage(org_img);
	IplImage * inverse_ = cvCreateImage(cvGetSize(org_), IPL_DEPTH_8U, 3);

	cvNot(org_, inverse_);

	inverse_img = cvarrToMat(inverse_);
	// display the source image
	imshow("Before", org_img);

	//smooth the image in the "src" and save it to "dst"
	blur(org_img, LQ_img1, Size(50, 50));
	blur(org_img, LQ_img2, Size(100, 100));
	blur(org_img, LQ_img3, Size(100, 100));
	blur(org_img, LQ_img4, Size(300, 300));
	blur(org_img, LQ_img5, Size(199, 199));
	blur(org_img, LQ_img6, Size(200, 200));
	blur(org_img, LQ_img7, Size(201, 201));
	blur(org_img, LQ_img8, Size(202, 202));
//	blur(inverse_img, LQ_inverse_img, Size(200, 200));
//	addWeighted(inverse_img, 1.5, LQ_inverse_img, -0.5, 0.0, HQ_inverse_img);
	
	//addWeighted(LQ_img1, 0.4, LQ_img2, 0.3, 0.0, LQ_img1);
	//addWeighted(LQ_img3, 0.2, LQ_img4, 0.1, 0.0, LQ_img3);
	addWeighted(LQ_img3, -1, LQ_img4, 1, 0.0, LQ_sum_img);
	addWeighted(LQ_img5, -100, LQ_img6, 100, 0.0, LQ_sum_img1);
	addWeighted(LQ_img7, -100, LQ_img8, 100, 0.0, LQ_sum_img2);
	

	IplImage *LQ_sum_ = new IplImage(LQ_sum_img);
	IplImage *LQ_sum_inverse_ = cvCreateImage(cvGetSize(LQ_sum_), IPL_DEPTH_8U, 3);

	cvNot(LQ_sum_, LQ_sum_inverse_);

	LQ_sum_inverse_img = cvarrToMat(LQ_sum_inverse_);

	addWeighted(org_img, 1, LQ_sum_img, 1, 0.0, sum_img);
	//addWeighted(org_img, 1, HQ_inverse_img, 0.3, 0.0, sum_img);

	//show the blurred image with the text
	//imshow("high qrequency inversed image", HQ_inverse_img);
	//imshow("Smoothing by avaraging", LQ_inverse_img);
	//imshow("Sum image", sum_img);
	imshow("Low freq _img ", LQ_sum_img);
	imshow("Low freq _img1 ", LQ_sum_img1);
	imshow("Low freq _img2 ", LQ_sum_img2);
	imshow("sum _img ", sum_img);
	imwrite(output_f, LQ_sum_img1);
	//wait for 3 seconds
	waitKey(60000);
}