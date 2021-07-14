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

#define output_f "out_pic.jpg"
# define th_ 8

using namespace cv;
using namespace std;


int under_f(int size, int num, int cnt) {
	if (size <= num) {
		cout << "size : " << size << endl;
		return cnt;
	}
	else
	{
		cnt++;
		cout << "size : " << size << endl;
		under_f(size / 2, num, cnt);
	}
}

uchar clipping(int pix_val) {
	int data;
	if (pix_val >= 255) {
		data = 255;
	}
	else if (pix_val <= 0) {
		data = 0;
	}
	else {
		data = pix_val;
	}
	return data;
}

Mat down_scale(Mat tmp, Mat down_img, int num) {
	for (num; num != 0; --num) {
		pyrDown(tmp, down_img, Size(tmp.cols / 2, tmp.rows / 2));
		tmp = down_img;
	}
	return down_img;
}

Mat up_scale(Mat tmp, Mat up_img, int num) {
	for (num; num != 0; --num) {
		pyrUp(tmp, up_img, Size(tmp.cols * 2, tmp.rows * 2));
		tmp = up_img;
	}
	return up_img;
}

/* Mat make_difference_info(Mat down_img, Mat diff_img, int th, InputArray xy_img) {

	int down_row, down_col;
	down_row = down_img.rows;
	down_col = down_img.cols;
	Mat xory_img = xy_img.getMat();

	uchar* pointer_output = diff_img.ptr<uchar>(0);
	pointer_output[0] = 0;
	pointer_output[1] = 0;
	pointer_output[2] = 0;

	for (int y = 0; y < down_row; y++) {
		uchar* pointer_input = down_img.ptr<uchar>(y);
		uchar* pointer_input_y = down_img.ptr<uchar>(y - 1);
		uchar* pointer_output = diff_img.ptr<uchar>(y);
		uchar* pointer_xy = xory_img.ptr<uchar>(y);

		if (y != 0) {
			for (int x = 0; x < down_col; x++) {

				if (x != 0) {
					int diff_b_x = -pointer_input[(x - 1) * 3 + 0] + pointer_input[x * 3 + 0];
					int diff_g_x = -pointer_input[(x - 1) * 3 + 1] + pointer_input[x * 3 + 1];
					int diff_r_x = -pointer_input[(x - 1) * 3 + 2] + pointer_input[x * 3 + 2];
					int diff_b_y = 0;//-pointer_input_y[x * 3 + 0] + pointer_input[x * 3 + 0];
					int diff_g_y = 0;// -pointer_input_y[x * 3 + 1] + pointer_input[x * 3 + 1];
					int diff_r_y = 0;// -pointer_input_y[x * 3 + 2] + pointer_input[x * 3 + 2];
					//uchar diff_b_xy = pointer_input_y[(x - 1) * 3 + 0] - pointer_input[x * 3 + 0];
					//uchar diff_g_xy = pointer_input_y[(x - 1) * 3 + 1] - pointer_input[x * 3 + 1];
					//uchar diff_r_xy = pointer_input_y[(x - 1) * 3 + 2] - pointer_input[x * 3 + 2];

					if (abs(diff_b_x) > th || abs(diff_b_y) > th) {
						if (abs(diff_b_x) > abs(diff_b_y)) {
							pointer_output[x * 3 + 0] = diff_b_x;

							pointer_xy[x * 3 + 0] = 0;
						}
						else {
							pointer_output[x * 3 + 0] = diff_b_y;
							pointer_xy[x * 3 + 0] = 1;
						}
					}
					else {
						pointer_output[x * 3 + 0] = 0;
						pointer_xy[x * 3 + 0] = 3;
					}

					if (abs(diff_g_x) > th || abs(diff_g_y) > th) {
						if (abs(diff_g_x) > abs(diff_g_y)) {
							pointer_output[x * 3 + 1] = diff_g_x;
							pointer_xy[x * 3 + 1] = 0;
						}
						else {
							pointer_output[x * 3 + 1] = diff_g_y;
							pointer_xy[x * 3 + 1] = 1;
						}
					}
					else {
						pointer_output[x * 3 + 1] = 0;
						pointer_xy[x * 3 + 1] = 3;
					}

					if (abs(diff_r_x) > th || abs(diff_r_y) > th) {
						if (abs(diff_r_x) > abs(diff_r_y)) {
							pointer_output[x * 3 + 2] = diff_r_x;
							pointer_xy[x * 3 + 2] = 0;
						}
						else {
							pointer_output[x * 3 + 2] = diff_r_y;
							pointer_xy[x * 3 + 2] = 1;
						}
					}
					else {
						pointer_output[x * 3 + 2] = 0;
						pointer_xy[x * 3 + 2] = 3;
					}
					
				}

				else {
					pointer_output[x * 3 + 0] = -pointer_input_y[x * 3 + 0] + pointer_input[x * 3 + 0];
					pointer_output[x * 3 + 1] = -pointer_input_y[x * 3 + 1] + pointer_input[x * 3 + 1];
					pointer_output[x * 3 + 2] = -pointer_input_y[x * 3 + 2] + pointer_input[x * 3 + 2];
					pointer_xy[x * 3 + 0] = 1;
					pointer_xy[x * 3 + 1] = 1;
					pointer_xy[x * 3 + 2] = 1;

				}

			}
		}
		else {
			for (int x = 1; x < down_col; x++) {

				pointer_output[0] = -pointer_input[(x - 1) * 3 + 0] + pointer_input[x * 3 + 0];
				pointer_output[1] = -pointer_input[(x - 1) * 3 + 1] + pointer_input[x * 3 + 1];
				pointer_output[2] = -pointer_input[(x - 1) * 3 + 2] + pointer_input[x * 3 + 2];
				pointer_xy[0] = 0;
				pointer_xy[1] = 0;
				pointer_xy[2] = 0;
			}

		}
		
	}
	imwrite(diff_f, diff_img);
	return  diff_img;
}
*/

Mat make_difference_x(Mat org_img, Mat diff_img, int th) {

	uchar* pointer_output_0 = diff_img.ptr<uchar>(0);
	pointer_output_0[0] = 0;
	pointer_output_0[1] = 0;
	pointer_output_0[2] = 0;
	int org_r = org_img.rows;
	int org_c = org_img.cols;

	for (int y = 0; y < org_r; y++) {
		uchar* pointer_input = org_img.ptr<uchar>(y);
		uchar* pointer_output = diff_img.ptr<uchar>(y);
		pointer_output[0] = 0;
		pointer_output[1] = 0;
		pointer_output[2] = 0;
		for (int x = 1; x < org_c; x++) {
			
				int diff_b_x = pointer_input[x * 3 + 0] - pointer_input[(x - 1) * 3 + 0];
				int diff_g_x = pointer_input[x * 3 + 1] - pointer_input[(x - 1) * 3 + 1];
				int diff_r_x = pointer_input[x * 3 + 2] - pointer_input[(x - 1) * 3 + 2];
				if (abs(diff_b_x > th)) pointer_output[x * 3 + 0] = diff_b_x;
				if (abs(diff_g_x > th)) pointer_output[x * 3 + 1] = diff_g_x;
				if (abs(diff_r_x > th)) pointer_output[x * 3 + 2] = diff_r_x;
				//..cout << (int)pointer_input[x * 3 + 0]  << " - " << (int)pointer_input[(x - 1) * 3 + 0] << " = " <<  diff_b_x << endl;
			}	
	}
	return  diff_img;
}

Mat make_difference_y(Mat org_img, Mat diff_img, int th) {

	uchar* pointer_output_0 = diff_img.ptr<uchar>(0);
	pointer_output_0[0] = 0;
	pointer_output_0[1] = 0;
	pointer_output_0[2] = 0;
	int org_r = org_img.rows;
	int org_c = org_img.cols;

	for (int y = 0; y < org_r; y++) {
		
		uchar* pointer_input = org_img.ptr<uchar>(y);
		uchar* pointer_input_y = org_img.ptr<uchar>(y - 1);
		uchar* pointer_output = diff_img.ptr<uchar>(y);
		if (y != 0) {
			for (int x = 0; x < org_c; x++) {

				int diff_b_y = pointer_input[x * 3 + 0] - pointer_input_y[(x) * 3 + 0];
				int diff_g_y = pointer_input[x * 3 + 1] - pointer_input_y[(x) * 3 + 1];
				int diff_r_y = pointer_input[x * 3 + 2] - pointer_input_y[(x) * 3 + 2];
				if (abs(diff_b_y > th)) pointer_output[x * 3 + 0] = diff_b_y;
				if (abs(diff_g_y > th)) pointer_output[x * 3 + 1] = diff_g_y;
				if (abs(diff_r_y > th)) pointer_output[x * 3 + 2] = diff_r_y;
			}
		}
		else {
			for (int x = 0; x < org_c; x++) {
				pointer_output[x * 3 + 0] = 0;
				pointer_output[x * 3 + 1] = 0;
				pointer_output[x * 3 + 2] = 0;
			}
		}
	}
	return  diff_img;
}

Mat make_white_img(Mat org_img, Mat diff_img, Mat xy_img) {
	Mat white_img(diff_img.rows, diff_img.cols, CV_8UC3);
	int diff_row = diff_img.rows;
	int diff_col = diff_img.cols;
	//up_img = up_scale(diff_img, up_img, cnt - 1);
	int* alpha_r = NULL;
	int* alpha_g = NULL;
	int* alpha_b = NULL;
	alpha_r = (int*)malloc(sizeof(int)*(diff_row*diff_col));
	alpha_g = (int*)malloc(sizeof(int)*(diff_row*diff_col));
	alpha_b = (int*)malloc(sizeof(int)*(diff_row*diff_col));

	*(alpha_b + 0) = (int)org_img.at<Vec3b>(0, 0)[0];
	*(alpha_g + 0) = (int)org_img.at<Vec3b>(0, 0)[1];
	*(alpha_r + 0) = (int)org_img.at<Vec3b>(0, 0)[2];

	for (int y = 0; y < diff_row; y++) {
		uchar* pointer_input = org_img.ptr<uchar>(y);
		uchar* pointer_diff = diff_img.ptr<uchar>(y);
		uchar* pointer_xy = xy_img.ptr<uchar>(y);
		uchar* pointer_output = white_img.ptr<uchar>(y);


		*(alpha_b + diff_row * y) = pointer_input[0];
		*(alpha_g + diff_row * y) = pointer_input[1];
		*(alpha_r + diff_row * y) = pointer_input[2];

		if (y == 0) {
			for (int x = 1; x < diff_col; x++) {
				uchar b_1 = pointer_input[x * 3 + 0];
				uchar g_1 = pointer_input[x * 3 + 1];
				uchar r_1 = pointer_input[x * 3 + 2];
				*(alpha_b + diff_row * y + x) = (pointer_diff[x * 3 + 0] > 125) ? (int)pointer_diff[x * 3 + 0] - 256 : (int)pointer_diff[x * 3 + 0];
				*(alpha_g + diff_row * y + x) = (pointer_diff[x * 3 + 1] > 125) ? (int)pointer_diff[x * 3 + 1] - 256 : (int)pointer_diff[x * 3 + 1];
				*(alpha_r + diff_row * y + x) = (pointer_diff[x * 3 + 2] > 125) ? (int)pointer_diff[x * 3 + 2] - 256 : (int)pointer_diff[x * 3 + 2];
				pointer_output[x * 3 + 0] = clipping((int)b_1 - *(alpha_b + diff_row * y + x));
				pointer_output[x * 3 + 1] = clipping((int)g_1 - *(alpha_g + diff_row * y + x));
				pointer_output[x * 3 + 2] = clipping((int)r_1 - *(alpha_r + diff_row * y + x));
			}
		}
		else {
			for (int x = 1; x < diff_col; x++) {
				uchar b_1 = pointer_input[x * 3 + 0];
				uchar g_1 = pointer_input[x * 3 + 1];
				uchar r_1 = pointer_input[x * 3 + 2];


				int temp_b = (pointer_diff[x * 3 + 0] > 125) ? (int)pointer_diff[x * 3 + 0] - 256 : (int)pointer_diff[x * 3 + 0];
				int temp_g = (pointer_diff[x * 3 + 1] > 125) ? (int)pointer_diff[x * 3 + 1] - 256 : (int)pointer_diff[x * 3 + 1];
				int temp_r = (pointer_diff[x * 3 + 2] > 125) ? (int)pointer_diff[x * 3 + 2] - 256 : (int)pointer_diff[x * 3 + 2];

				//*(alpha_b + diff_row * y + x) = temp_b;
				//*(alpha_g + diff_row * y + x) = temp_g;
				//*(alpha_r + diff_row * y + x) = temp_r;

				*(alpha_b + diff_row * y + x) = *(alpha_b + diff_row * y + (x - 1)) + (*(alpha_b + diff_row * (y - 1) + (x)) - *(alpha_b + diff_row * (y - 1) + (x - 1))) + temp_b;
				*(alpha_g + diff_row * y + x) = *(alpha_g + diff_row * y + (x - 1)) + (*(alpha_g + diff_row * (y - 1) + (x)) - *(alpha_g + diff_row * (y - 1) + (x - 1))) + temp_g;
				*(alpha_r + diff_row * y + x) = *(alpha_r + diff_row * y + (x - 1)) + (*(alpha_r + diff_row * (y - 1) + (x)) - *(alpha_r + diff_row * (y - 1) + (x - 1))) + temp_r;

				//cout << "alpha_b : " << *(alpha_b + diff_row * y + x) << endl;
				//cout << "alpha_g : " << *(alpha_g + diff_row * y + x) << endl;
				//cout << "alpha_r : " << *(alpha_r + diff_row * y + x) << endl;

				
				//cout << (int)b_1 - *(alpha_b + diff_row * y + x) << "  " << (int)clipping((int)b_1 - *(alpha_b + diff_row * y + x)) << endl;
				//cout << (int)b_1 << " - " << (int)*(alpha_b + diff_row * y + x) << " = "  << (int)pointer_output[x * 3 + 0] << endl;
				//cout << "alpha_g : " << *(alpha_g + diff_row * y + x) << endl;
				//cout << "alpha_r : " << *(alpha_r + diff_row * y + x) << endl;
				pointer_output[x * 3 + 0] = clipping((int)b_1 - *(alpha_b + diff_row * y + x));
				pointer_output[x * 3 + 1] = clipping((int)g_1 - *(alpha_g + diff_row * y + x));
				pointer_output[x * 3 + 2] = clipping((int)r_1 - *(alpha_r + diff_row * y + x));

			}
		}
		
	}
	return white_img;
}

Mat make_white_img_x(Mat org_img, Mat diff_img) {
	Mat white_img_x(diff_img.rows, diff_img.cols, CV_8UC3);
	int org_row = org_img.rows;
	int org_col = org_img.cols;
	int diff_row = diff_img.rows;
	int diff_col = diff_img.cols;

	int* alpha_r = (int*)malloc(sizeof(int)*(org_row*org_col));
	int* alpha_g = (int*)malloc(sizeof(int)*(org_row*org_col));
	int* alpha_b = (int*)malloc(sizeof(int)*(org_row*org_col));

	*(alpha_b + 0) = 0;
	*(alpha_g + 0) = 0;
	*(alpha_r + 0) = 0;

	for (int y = 0; y < org_row; y++) {
		uchar* pointer_input = org_img.ptr<uchar>(y);
		uchar* pointer_diff = diff_img.ptr<uchar>(y);
		uchar* pointer_output = white_img_x.ptr<uchar>(y);
		*(alpha_b + diff_row * y) = pointer_input[0];
		*(alpha_g + diff_row * y) = pointer_input[1];
		*(alpha_r + diff_row * y) = pointer_input[2];
		for (int x = 1; x < diff_col; x++) {

			uchar b_1 = pointer_input[x * 3 + 0];
			uchar g_1 = pointer_input[x * 3 + 1];
			uchar r_1 = pointer_input[x * 3 + 2];

			*(alpha_b + diff_row * y + x) = *(alpha_b + diff_row * y + (x - 1)) + ((pointer_diff[x * 3 + 0] > 125) ? (int)pointer_diff[x * 3 + 0] - 256 : (int)pointer_diff[x * 3 + 0]);
			*(alpha_g + diff_row * y + x) = *(alpha_g + diff_row * y + (x - 1)) + ((pointer_diff[x * 3 + 1] > 125) ? (int)pointer_diff[x * 3 + 1] - 256 : (int)pointer_diff[x * 3 + 1]);
			*(alpha_r + diff_row * y + x) = *(alpha_r + diff_row * y + (x - 1)) + ((pointer_diff[x * 3 + 2] > 125) ? (int)pointer_diff[x * 3 + 2] - 256 : (int)pointer_diff[x * 3 + 2]);

			pointer_output[x * 3 + 0] = clipping((int)b_1 - *(alpha_b + diff_row * y + x));
			pointer_output[x * 3 + 1] = clipping((int)g_1 - *(alpha_g + diff_row * y + x));
			pointer_output[x * 3 + 2] = clipping((int)r_1 - *(alpha_r + diff_row * y + x));
		//	cout << (int)pointer_diff[x * 3 + 0] << "    " <<(int) *(alpha_b + diff_row * y + x) << endl;
		}

	}
	free(alpha_b);
	free(alpha_g);
	free(alpha_r);
	return white_img_x;
}

Mat make_white_img_y(Mat org_img, Mat diff_img_y) {
	Mat white_img_y(diff_img_y.rows, diff_img_y.cols, CV_8UC3);
	int org_row = org_img.rows;
	int org_col = org_img.cols;
	int diff_row = diff_img_y.rows;
	int diff_col = diff_img_y.cols;

	int* alpha_r = (int*)malloc(sizeof(int)*(org_row*org_col));
	int* alpha_g = (int*)malloc(sizeof(int)*(org_row*org_col));
	int* alpha_b = (int*)malloc(sizeof(int)*(org_row*org_col));

	*(alpha_b + 0) = 0;
	*(alpha_g + 0) = 0;
	*(alpha_r + 0) = 0;

	for (int y = 0; y < org_row; y++) {

		uchar* pointer_input = org_img.ptr<uchar>(y);
		uchar* pointer_diff = diff_img_y.ptr<uchar>(y);
		uchar* pointer_output = white_img_y.ptr<uchar>(y);
		if (y == 0) {
			for (int x = 0; x < diff_col; x++) {
				uchar b_1 = pointer_input[x * 3 + 0];
				uchar g_1 = pointer_input[x * 3 + 1];
				uchar r_1 = pointer_input[x * 3 + 2];

				*(alpha_b + diff_row * y + x) = (pointer_diff[x * 3 + 0] > 125) ? (int)pointer_diff[x * 3 + 0] - 256 : (int)pointer_diff[x * 3 + 0];
				*(alpha_g + diff_row * y + x) = (pointer_diff[x * 3 + 1] > 125) ? (int)pointer_diff[x * 3 + 1] - 256 : (int)pointer_diff[x * 3 + 1];
				*(alpha_r + diff_row * y + x) = (pointer_diff[x * 3 + 2] > 125) ? (int)pointer_diff[x * 3 + 2] - 256 : (int)pointer_diff[x * 3 + 2];

				pointer_output[x * 3 + 0] = clipping((int)b_1 - *(alpha_b + diff_row * y + x));
				pointer_output[x * 3 + 1] = clipping((int)g_1 - *(alpha_g + diff_row * y + x));
				pointer_output[x * 3 + 2] = clipping((int)r_1 - *(alpha_r + diff_row * y + x));
			}
		}
		else {
			for (int x = 0; x < diff_col; x++) {

				uchar b_1 = pointer_input[x * 3 + 0];
				uchar g_1 = pointer_input[x * 3 + 1];
				uchar r_1 = pointer_input[x * 3 + 2];

				*(alpha_b + diff_row * y + x) = *(alpha_b + diff_row * (y - 1) + x) + ((pointer_diff[x * 3 + 0] > 125) ? (int)pointer_diff[x * 3 + 0] - 256 : (int)pointer_diff[x * 3 + 0]);
				*(alpha_g + diff_row * y + x) = *(alpha_g + diff_row * (y - 1) + x) + ((pointer_diff[x * 3 + 1] > 125) ? (int)pointer_diff[x * 3 + 1] - 256 : (int)pointer_diff[x * 3 + 1]);
				*(alpha_r + diff_row * y + x) = *(alpha_r + diff_row * (y - 1) + x) + ((pointer_diff[x * 3 + 2] > 125) ? (int)pointer_diff[x * 3 + 2] - 256 : (int)pointer_diff[x * 3 + 2]);

				pointer_output[x * 3 + 0] = clipping((int)b_1 - *(alpha_b + diff_row * y + x));
				pointer_output[x * 3 + 1] = clipping((int)g_1 - *(alpha_g + diff_row * y + x));
				pointer_output[x * 3 + 2] = clipping((int)r_1 - *(alpha_r + diff_row * y + x));
			//	cout << (int)b_1 << endl;
			}
		}
	}
	free(alpha_b);
	free(alpha_g);
	free(alpha_r);
	return white_img_y;
}

Mat weighted(Mat org_img, int weight, int shift) {
	Mat weighted_img;
	int height = org_img.rows;
	int width = org_img.cols;
	for (int y = 0; y < height; y++) {
		uchar* pointer_input = org_img.ptr<uchar>(y);
		uchar* pointer_output = weighted_img.ptr<uchar>(y);
			for (int x = 0; x < width; x++) {
				pointer_output[x * 3 + 0] = clipping((int)pointer_input[x * 3 + 0] * weight + shift);
				pointer_output[x * 3 + 1] = clipping((int)pointer_input[x * 3 + 1] * weight + shift);
				pointer_output[x * 3 + 2] = clipping((int)pointer_input[x * 3 + 2] * weight + shift);
			}
		
	}
	return weighted_img;
}

int main(int argc, char** argv)
{
	// Load the source image
	Mat org_img = imread(input_f, 1);
	Mat LQ_img;
	Mat tmp,down_img,up_img;
	tmp = org_img;
	down_img = tmp;
	
	int height_org = org_img.rows;
	int width_org = org_img.cols;
	int scale_st = 512;
	int cnt;
	
	
	blur(org_img, org_img, Size(18, 18));
	//re_img = weighted(LQ_img, 0.5, 0);
	//cnt = min(under_f(org_img.rows, scale_st, 1), under_f(org_img.cols, scale_st, 1));

	//down_img = down_scale(tmp, down_img, cnt-1);
	//cout << "down_img.rows : " << down_img.rows << endl;
	//cout << "down_img.cols : " << down_img.cols << endl;
	

	Mat diff_img_x(org_img.rows, org_img.cols, CV_8UC3);
	Mat diff_img_y(org_img.rows, org_img.cols, CV_8UC3);
	Mat white_img_x(height_org, width_org, CV_8UC3);
	Mat white_img_y(height_org, width_org, CV_8UC3);
	Mat white_img(height_org, width_org, CV_8UC3);

	diff_img_x = make_difference_x(org_img, diff_img_x, th_);
	diff_img_y = make_difference_y(org_img, diff_img_y, th_);
	white_img_x = make_white_img_x(org_img, diff_img_x);
	white_img = make_white_img_y(white_img_x, diff_img_y);
	
	

	imshow("White image", white_img);
	
	imwrite(output_f, white_img_x);

	//wait for 3 seconds
	waitKey(60000);
}
