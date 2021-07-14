#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui_c.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define input_f "pic.jpg"
#define diff_f "diff.jpg"
#define k_size  10
#define scale  1//1 or 3
#define delta  0
#define ksize 3//1 or 3
#define output_D_f "out_picD.jpg"
#define output_W_f "out_picW.jpg"
#define diff_th_D 0
#define diff_th_W_around 20
#define diff_th_W_cur 100
#define clipping_th_ 5
#define blur_size 10
//#define gab 5
#define around_type 4

using namespace cv;
using namespace std;


Mat remove_noise(InputArray org_img) {
	Mat _org_img = org_img.getMat();
	Mat erode_img(_org_img.rows, _org_img.cols, CV_8UC3);
	Mat dilation_img(_org_img.rows, _org_img.cols, CV_8UC3);
	Mat kernel(k_size, k_size, CV_8UC1, Scalar(255));
	erode(_org_img, erode_img, kernel);
	dilate(erode_img, dilation_img, kernel);
	return dilation_img;

}

void Mat2intarr(InputArray diff_img, int* diff_info) {
	Mat _diff_img = diff_img.getMat();
	int height_org = _diff_img.rows;
	int width_org = _diff_img.cols;
	for (int i = 0; i < height_org; i++) {
		for (int j = 0; j < width_org; j++) {
			*(diff_info + width_org * i + j) = (int)(_diff_img.at<uchar>(i, j)) - 125;
			//   cout << (int)_diff_img.at<uchar>(i, j) << endl;
		}
	}
}

void floatarr2Mat(float* diff_info, OutputArray diff_img) {
	Mat _diff_img = diff_img.getMat();
	int height_org = _diff_img.rows;
	int width_org = _diff_img.cols;
	for (int i = 0; i < height_org; i++) {
		for (int j = 0; j < width_org; j++) {
			(_diff_img.at<uchar>(i, j)) = ((uchar)(*(diff_info + width_org * i + j)));
		}
	}
}

void Mat2floatarr(InputArray Mat_img, float* f_img) {
	Mat _Mat_img = Mat_img.getMat();
	int height_org = _Mat_img.rows;
	int width_org = _Mat_img.cols;
	for (int i = 0; i < height_org; i++) {
		for (int j = 0; j < width_org; j++) {
			*(f_img + width_org * i + j) = (float)(_Mat_img.at<uchar>(i, j));
		}
	}
}

int clipping(int pix_val, int ddepth) {
	int data;
	if (ddepth == CV_8U) {
		if (pix_val >= 255) {
			data = 255;
		}
		else if (pix_val <= 0) {
			data = 0;
		}
		else {
			data = pix_val;
		}
	}
	else if (ddepth == CV_8S) {
		if (pix_val >= 127) {
			data = 127;
		}
		else if (pix_val <= -128) {
			data = 127;
		}
		else {
			data = pix_val;
		}
	}
	else if (ddepth == CV_16U) {
		if (pix_val >= 65535) {
			data = 65535;
		}
		else if (pix_val <= 0) {
			data = 0;
		}
		else {
			data = pix_val;
		}
	}
	else if (ddepth == CV_16S) {
		if (pix_val >= 32767) {
			data = 32767;
		}
		else if (pix_val <= -32768) {
			data = -32768;
		}
		else {
			data = pix_val;
		}
	}
	return data;
}

float clipping_abs(float f_val, int threshold) {
	float data;
	int i_val = (int)f_val;
	if (abs(f_val) >= threshold) {
		data = (float)threshold;
	}
	else {
		data = f_val;
	}
	return data;
}

Mat findMax_adjustment(InputArray Gray_img) {
	Mat _Gray_img = Gray_img.getMat();
	Mat dst_img(_Gray_img.rows, _Gray_img.cols, CV_16SC1);
	int height_org = _Gray_img.rows;
	int width_org = _Gray_img.cols;

	int max = _Gray_img.at<short>(0, 0);
	int min = _Gray_img.at<short>(0, 0);
	for (int i = 0; i < height_org; i++) {
		for (int j = 0; j < width_org; j++) {
			if (max < _Gray_img.at<short>(i, j)) {
				max = _Gray_img.at<short>(i, j);
			}
			else if (min > _Gray_img.at<short>(i, j)) {
				min = _Gray_img.at<short>(i, j);
			}
		}
	}
	int temp_max = max - 255;//0보다 작으면 이미지가 255 위로 넘어가지 않음
	int temp_min = -min;//0보다 작으면 이미지가 0 아래로 넘어가지 않음

	if (temp_max > 0 && temp_min < 0) {
		for (int i = 0; i < height_org; i++) {
			for (int j = 0; j < width_org; j++) {
				dst_img.at<short>(i, j) = _Gray_img.at<short>(i, j) - temp_max;
			}
		}
		return dst_img;
	}
	else if (temp_max < 0 && temp_min > 0) {
		for (int i = 0; i < height_org; i++) {
			for (int j = 0; j < width_org; j++) {
				dst_img.at<short>(i, j) = _Gray_img.at<short>(i, j) + temp_min;
			}
		}
		return dst_img;
	}
	else if (temp_max < 0 && temp_min > 0) {
		Mat abs_Gray_img;
		convertScaleAbs(_Gray_img, abs_Gray_img);
		return dst_img;
	}
	else {
		return _Gray_img;
	}
}

Mat diff2_xy_func(InputArray Gray_img, int depth_) {
	Mat _Gray_img = Gray_img.getMat();
	Mat _2diff_xy_img(_Gray_img.rows, _Gray_img.cols, CV_16SC1);// = grad_xy_img.getMat();
	float data[] = { 1,-1,0,  -1,1,0,  0,0,0 };
	Mat kernel(3, 3, CV_32F, data);
	filter2D(_Gray_img, _2diff_xy_img, depth_, kernel, Point(-1, -1), 0, BORDER_CONSTANT);
	return  _2diff_xy_img;
}

Mat Laplacian_func(InputArray Gray_img, int depth_) {
	Mat _Gray_img = Gray_img.getMat();
	Mat _Laplacian_img(_Gray_img.rows, _Gray_img.cols, CV_16SC1);// = grad_xy_img.getMat();
	float data[] = { 0,-1,0,  -1,4, -1 ,  0,-1,0 };
	Mat kernel(5, 5, CV_32F, data);
	filter2D(_Gray_img, _Laplacian_img, depth_, kernel, Point(-1, -1), 0, BORDER_CONSTANT);
	return  _Laplacian_img;
}

Mat diff_sum_1D_x(InputArray Gray_img, int depth_) {
	Mat _Gray_img = Gray_img.getMat();
	Mat _1diff_x_img(_Gray_img.rows, _Gray_img.cols, CV_16SC1);// = grad_xy_img.getMat();
	float data[] = { 0,0,0,0,0,0,0, 0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  1,1,1,1,1,1,1,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0, 0,0,0,0,0,0,0, };
	Mat kernel(7, 7, CV_32F, data);
	filter2D(_Gray_img, _1diff_x_img, depth_, kernel, Point(-1, -1), 0, BORDER_CONSTANT);
	return  _1diff_x_img;
}

Mat diff_sum_1D_y(InputArray Gray_img, int depth_) {
	Mat _Gray_img = Gray_img.getMat();
	Mat _1diff_y_img(_Gray_img.rows, _Gray_img.cols, CV_16SC1);// = grad_xy_img.getMat();
	float data[] = { 0,0,0,1,0,0,0, 0,0,0,1,0,0,0,  0,0,0,1,0,0,0  ,0,0,0,1,0,0,0,  0,0,0,1,0,0,0,  0,0,0,1,0,0,0,  0,0,0,1,0,0,0 };
	Mat kernel(7, 7, CV_32F, data);
	filter2D(_Gray_img, _1diff_y_img, depth_, kernel, Point(-1, -1), 0, BORDER_CONSTANT);
	return  _1diff_y_img;
}

Mat diff_sum_2D_xy(InputArray Gray_img, int depth_) {
	Mat _Gray_img = Gray_img.getMat();
	Mat _1diff_y_img(_Gray_img.rows, _Gray_img.cols, CV_16SC1);// = grad_xy_img.getMat();
	float data[] = { 1,0,0,0,0,0,0, 0,1,0,0,0,0,0,  0,0,1,0,0,0,0  ,0,0,0,1,0,0,0,  0,0,0,0,1,0,0,  0,0,0,0,0,1,0,  0,0,0,0,0,0,1 };
	Mat kernel(7, 7, CV_32F, data);
	filter2D(_Gray_img, _1diff_y_img, depth_, kernel, Point(-1, -1), 0, BORDER_CONSTANT);
	return  _1diff_y_img;
}

Mat diff_sum_2D_yx(InputArray Gray_img, int depth_) {
	Mat _Gray_img = Gray_img.getMat();
	Mat _1diff_y_img(_Gray_img.rows, _Gray_img.cols, CV_16SC1);// = grad_xy_img.getMat();
	float data[] = { 0,0,0,0,0,0,1, 0,0,0,0,0,1,0,  0,0,0,0,1,0,0  ,0,0,0,1,0,0,0,  0,0,1,0,0,0,0,  0,1,0,0,0,0,0,  1,0,0,0,0,0,0 };
	Mat kernel(7, 7, CV_32F, data);
	filter2D(_Gray_img, _1diff_y_img, depth_, kernel, Point(-1, -1), 0, BORDER_CONSTANT);
	return  _1diff_y_img;
}

void max_min_info(int* around, int* max_val, int* max_index, int*min_val, int* min_index) {
	*max_val = *(around + 0);
	*min_val = *(around + 0);
	*min_index = 0;
	*max_index = 0;
	for (int i = 0; i < around_type; i++) {
		if (*max_val < *(around + i)) {
			*max_val = *(around + i);
			*(max_index) = i;
		}
		if (*min_val > *(around + i)) {
			*min_val = *(around + i);
			*(min_index) = i;
		}
	}
}

Mat Cumulative_Sum_diff2_s(InputArray Gray_img, InputArray diff_img, int threshVal_around, int threshVal_cur) {
	Mat _Gray_img = Gray_img.getMat();
	Mat _diff_img = diff_img.getMat();
	Mat _sum_img(_diff_img.rows, _diff_img.cols, CV_16SC1);
	Mat adjusted_sum_img(_diff_img.rows, _diff_img.cols, CV_16SC1);
	Mat diff_sum1_x(_diff_img.rows, _diff_img.cols, CV_16SC1);
	Mat diff_sum1_y(_diff_img.rows, _diff_img.cols, CV_16SC1);
	Mat diff_sum2_xy(_diff_img.rows, _diff_img.cols, CV_16SC1);
	Mat diff_sum2_yx(_diff_img.rows, _diff_img.cols, CV_16SC1);
	diff_sum1_x = diff_sum_1D_x(_diff_img, CV_16S);
	diff_sum1_y = diff_sum_1D_y(_diff_img, CV_16S);
	diff_sum2_xy = diff_sum_2D_xy(_diff_img, CV_16S);
	diff_sum2_yx = diff_sum_2D_yx(_diff_img, CV_16S);

	int height_org = _sum_img.rows;
	int width_org = _sum_img.cols;
	int temp = 0;
	_sum_img.at<short>(0, 0) = clipping(_diff_img.at<short>(0, 0), CV_16S);
	for (int i = 0; i < height_org; i++) {
		for (int j = 0; j < width_org; j++) {

			if (!(i == 0 && j == 0)) {

				if (i == 0) {
					temp = _sum_img.at<short>(i, j - 1);
				}
				else if (j == 0) {
					temp = _sum_img.at<short>(i - 1, j);
				}
				else {
					temp = _sum_img.at<short>(i - 1, j) + ((int)_sum_img.at<short>(i, j - 1) - (int)_sum_img.at<short>(i - 1, j - 1));
				}

				int around_diff[4] = { abs(diff_sum1_x.at<short>(i, j)) , abs(diff_sum1_y.at<short>(i, j)) ,abs(diff_sum2_xy.at<short>(i, j)) ,abs(diff_sum2_yx.at<short>(i, j)) };
				int around_max_value, max_index, around_min_value, min_index;
				max_min_info(around_diff, &around_max_value, &max_index, &around_min_value, &min_index);

				if (abs(around_max_value) <= threshVal_around)_sum_img.at<short>(i, j) = clipping(temp + (int)_diff_img.at<short>(i, j), CV_16S);
				else {

					if (min_index == 0) {
						_sum_img.at<short>(i, j) = (j != 0) ? _sum_img.at<short>(i, j-1): temp;
					}
					else if (min_index == 1) {
						_sum_img.at<short>(i, j) = (i != 0) ? _sum_img.at<short>(i-1, j): temp;
					}

					else if (min_index == 2) {
						_sum_img.at<short>(i, j) = (j != 0 && i != 0) ? _sum_img.at<short>(i-1, j-1): temp;
					}
					else if (min_index == 3) {
						_sum_img.at<short>(i, j) = (j != width_org - 1 && i != 0) ? _sum_img.at<short>(i-1, j+1): temp;
					}

				}
			}
		}
	}
	cout << endl;//test
	//adjusted_sum_img = findMax_adjustment(_sum_img);
	return _sum_img;
}

Mat subtract_diff(InputArray Gray_img, InputArray Sum_img) {
	Mat _Gray_img = Gray_img.getMat();
	Mat _Sum_img = Sum_img.getMat();
	Mat sub_Gray_img(_Gray_img.rows, _Gray_img.cols, CV_16SC1);
	int height_org = _Gray_img.rows;
	int width_org = _Gray_img.cols;
	for (int i = 0; i < height_org; i++) {
		for (int j = 0; j < width_org; j++) {
			sub_Gray_img.at<uchar>(i, j) = clipping(_Gray_img.at<short>(i, j) - _Sum_img.at<short>(i, j), CV_16S);
		}
	}
	return sub_Gray_img;
}

Mat Cumulative_Sum_diff2_f(InputArray Gray_img, InputArray diff_img, int threshVal) {


	Mat _Gray_img = Gray_img.getMat();
	Mat _diff_img = diff_img.getMat();
	Mat _sum_img(_diff_img.rows, _diff_img.cols, CV_16SC1);
	int height_org = _sum_img.rows;
	int width_org = _sum_img.cols;

	int temp = 0;
	_sum_img.at<short>(0, 0) = clipping(_diff_img.at<float>(0, 0), CV_16S);

	for (int i = 0; i < height_org; i++) {

		for (int j = 0; j < width_org; j++) {
			if (!(i == 0 && j == 0)) {


				if (i == 0) {
					temp = _sum_img.at<short>(i, j - 1);
				}
				else if (j == 0) {
					temp = _sum_img.at<short>(i - 1, j);
				}
				else {
					temp = _sum_img.at<short>(i - 1, j) + ((int)_sum_img.at<short>(i, j - 1) - (int)_sum_img.at<short>(i - 1, j - 1));
				}

				if (abs((int)(_diff_img.at<float>(i, j))) > threshVal) _sum_img.at<short>(i, j) = clipping(temp + (int)_diff_img.at<float>(i, j), CV_16S);
				else _sum_img.at<short>(i, j) = clipping(temp, CV_16S);
				//   cout << _diff_img.at<float>(i, j) << endl;
			}
		}
	}


	return _sum_img;
}

void div_image(InputArray A_img, InputArray B_img, float* div_info) {
	Mat _A_img = A_img.getMat();
	Mat _B_img = B_img.getMat();

	int height_org = _A_img.rows;
	int width_org = _A_img.cols;
	for (int i = 0; i < height_org; i++) {
		for (int j = 0; j < width_org; j++) {
			*(div_info + i * width_org + j) = clipping((float)_A_img.at<uchar>(i, j) / (float)_B_img.at<uchar>(i, j), CV_8U);
			//cout << *(div_info + i * width_org + j) << endl;
		}
	}
}

Mat sobel_xy_func(InputArray Gray_img, int _ddepth, int _ksize, int _scale, int _delta) {
	Mat _Gray_img = Gray_img.getMat();
	Mat _grad_xy_img(_Gray_img.rows, _Gray_img.cols, CV_16SC1);// = grad_xy_img.getMat();
	Mat grad_x;
	Sobel(_Gray_img, grad_x, _ddepth, 1, 0, _ksize, _scale, _delta, BORDER_DEFAULT);
	Sobel(grad_x, _grad_xy_img, _ddepth, 0, 1, _ksize, _scale, _delta, BORDER_DEFAULT);
	return _grad_xy_img;
}

Mat div_image_f(InputArray Laplacian_img, InputArray grad_xy_img) {
	Mat _Laplacian_img = Laplacian_img.getMat();
	Mat _grad_xy_img = grad_xy_img.getMat();

	int height_org = _grad_xy_img.rows;
	int width_org = _grad_xy_img.cols;
	Mat _div_float_img(height_org, width_org, CV_32FC1);

	for (int i = 0; i < height_org; i++) {
		for (int j = 0; j < width_org; j++) {
			float temp_f = ((_Laplacian_img.at<short>(i, j)) == 0) ? 0.2 : ((_Laplacian_img.at<short>(i, j))*(-1));
			_div_float_img.at<float>(i, j) = (float)_grad_xy_img.at<short>(i, j) / temp_f;
			//cout << _div_float_img.at<float>(i, j) << endl;
		}
	}
	return _div_float_img;
}

Mat extraction_depth(InputArray org_img, int _th_D) {
	Mat _org_img = org_img.getMat();
	int height_org = _org_img.rows;
	int width_org = _org_img.cols;
	Mat _gradient_D_img(height_org, width_org, CV_16SC1);

	Mat div_float_img(height_org, width_org, CV_32FC1);
	Mat Gray_img(height_org, width_org, CV_8UC1);
	Mat Laplacian_img(height_org, width_org, CV_16SC1);
	Mat grad_xy_img(height_org, width_org, CV_16SC1);
	int ddepth = CV_16S;

	cvtColor(_org_img, Gray_img, CV_BGR2GRAY);
	Laplacian_img = Laplacian_func(Gray_img, ddepth); //I 를 x와 y 에 대해 라플라시안
	grad_xy_img = diff2_xy_func(Gray_img, ddepth);//I 를 x,y 에 대해 이중 미분
	div_float_img = div_image_f(Laplacian_img, grad_xy_img); // I의 라플라시안 값과 I 의 이중 미분값을 나눠 줌
	_gradient_D_img = Cumulative_Sum_diff2_f(Gray_img, div_float_img, _th_D);//x,y 에 이중 적분

	return  _gradient_D_img;
}

Mat  extraction_white(InputArray org_img, int _th_W_around, int _th_W_cur) {
	Mat _org_img = org_img.getMat();
	Mat _white_img(_org_img.rows, _org_img.cols, CV_8UC3);
	int height_org = _org_img.rows;
	int width_org = _org_img.cols;
	int ddepth = CV_16S;
	vector<Mat> bgr_img(3);
	vector<Mat> remove_noise_img(3);
	vector<Mat> grad_xy_img(3);
	vector<Mat> diff_img(3);
	vector<Mat> W_img(3);
	split(_org_img, bgr_img);//0 : B, 1 : G, 2 : R
	int index;
	for (index = 0; index < 3; index++) {
		grad_xy_img[index] = diff2_xy_func(bgr_img[index], ddepth);
		diff_img[index] = Cumulative_Sum_diff2_s(bgr_img[index], grad_xy_img[index], _th_W_around, _th_W_cur);
		W_img[index] = subtract_diff(bgr_img[index], diff_img[index]);
	}
	merge(diff_img, _white_img);
	//   imwrite("merge.jpg", _white_img);//test ================
	return _white_img;
}

int main(int argc, char** argv)
{
	// Load the source image
	Mat org_img = imread(input_f, 1);
	Mat abs_gradient_D_img, abs_gradient_W_img;//test
	Mat gradient_D_img(org_img.rows, org_img.cols, CV_16SC1);
	Mat white_img(org_img.rows, org_img.cols, CV_8UC3);
	Mat blur_img(org_img.rows, org_img.cols, CV_8UC3);
	blur(org_img, blur_img, Size(blur_size, blur_size));
	//blur_img = remove_noise(org_img);
	//imwrite("removed_img.jpg", blur_img);
	//gradient_D_img = extraction_depth(blur_img, diff_th_D);//depth 추출
	//convertScaleAbs(gradient_D_img, abs_gradient_D_img);
	//imwrite(output_D_f, gradient_D_img);//출력 이미지
	white_img = extraction_white(blur_img, diff_th_W_around, diff_th_W_cur);
	imwrite(output_W_f, white_img);//출력 이미지

							//wait for 100 seconds
	waitKey(100000);

	return 0;
}