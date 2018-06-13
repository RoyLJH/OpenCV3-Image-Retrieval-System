// Image_Retrieval.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <windows.h>
#include <memory.h>
#include <math.h>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

#define MAX_PICNNUM  15000
#define TRAINING_MODE  0
#define TEST_MODE      1
Mat UserChosenMat;
double max_glcm_feature[4];
double min_glcm_feature[4];

std::vector<std::string> split(std::string str, std::string pattern)
{
	     std::string::size_type pos;
	     std::vector<std::string> result;
	     str += pattern;//扩展字符串以方便操作
	     int size = str.size();
	
		     for (int i = 0; i<size; i++)
		     {
		         pos = str.find(pattern, i);
		         if (pos<size)
			         {
			             std::string s = str.substr(i, pos - i);
			             result.push_back(s);
			             i = pos + pattern.size() - 1;
			         }
		     }
	     return result;
}

string getPath(int picID) {
	return "E:\\Image Lib\\mirflickr25k\\mirflickr\\im" + to_string(picID) + ".jpg";
}
void showPic(int picID, string windowName) {
	namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	Mat show = imread(getPath(picID), CV_LOAD_IMAGE_COLOR);
	imshow(windowName, show);
}
string Color_PHash(int srcPicNum, int mode)
{
	Mat src;
	if (mode == TRAINING_MODE) {
		string srcPath = getPath(srcPicNum);
		src = imread(srcPath, CV_LOAD_IMAGE_COLOR);
	}
	else {
		src = UserChosenMat;
	}
	Mat img, dst;
	string rst(64, '\0');
	double dIdex[64];
	double mean = 0.0;
	int k = 0;
	if (src.channels() == 3)
	{
		cvtColor(src, src, CV_BGR2GRAY);
		img = Mat_<double>(src);
	}
	else
	{
		img = Mat_<double>(src);
	}
	//1.Resize to 8*8
	resize(img, img, Size(8, 8));
	//2.dct
	dct(img, dst);
	//3.求左上角dct系数 8*8
	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j)
		{
			dIdex[k] = dst.at<double>(i, j);
			mean += dst.at<double>(i, j) / 64;
			++k;
		}
	}
	//4.calculate Hash
	for (int i = 0; i<64; ++i)
	{
		if (dIdex[i] >= mean)
		{
			rst[i] = '1';
		}
		else
		{
			rst[i] = '0';
		}
	}

	return rst;
}
string Color_hsv_hist(int srcPicNum, int mode) {
	Mat src, hsv;
	int bin[64];
	//8*h + s
	for (int i = 0; i < 64; i++) {
		bin[i] = 0;
	}
	if (mode == TRAINING_MODE) {
		src = imread(getPath(srcPicNum), CV_LOAD_IMAGE_COLOR);
	}
	else {
		src = UserChosenMat;
	}
	resize(src, src, Size(45, 45));
	cvtColor(src, hsv, COLOR_RGB2HSV);
	vector<cv::Mat> hsv_vec;
	split(hsv, hsv_vec);
	Mat H = hsv_vec[0];
	Mat S = hsv_vec[1];
	H.convertTo(H, CV_32F);
	S.convertTo(S, CV_32F);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			float h = H.at<float>(i, j);
			float s = S.at<float>(i, j);
			//cout << "h:" << h << ",s:" << s << endl;
			int hi = ((int)h-1) / 45 * 2;
			int yy = ((int)h - 1) % 45;
			hi = (yy >= 22) ? (hi + 1) : hi;
			int si = ((int)s - 1) / 32;
			int bidx = hi * 8 + si;
			bin[bidx]++;
		}
	}
	string hsv_code;
	for (int i = 0; i < 64; i++) {
		hsv_code.append(to_string(bin[i]));
		if (i != 63)	hsv_code.push_back(',');
	}
	//cout << hsv_code << endl;
	//string windowName = "tests";
	//namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	//imshow(windowName, hsv);
	return hsv_code;
}
string Color_otsu(int srcPicNum, int mode) {
	Mat src;
	if (mode == 0) {
		src = imread(getPath(srcPicNum), CV_LOAD_IMAGE_COLOR);
	}
	else {
		src = UserChosenMat;
	}
	resize(src,src,Size(40,40));
	Mat gray, dst;
	cvtColor(src, gray, CV_RGB2GRAY);
	threshold(gray, dst, 0, 255, CV_THRESH_OTSU);
	string otsu(1600,'\0');
	for (int i = 0; i < 40; i++) {
		for (int j = 0; j < 40; j++) {
			int k = (int)(dst.at<char>(i, j)) + 1;
			if (k == 1)
				otsu[i * 40 + j] = '1';
			else
				otsu[i * 40 + j] = '0';
		}
	}

	//string windowName = "tests";
	//namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	//imshow(windowName, dst);
	return otsu;
}
string Texture_GLCM(int srcPicNum, int mode) {
	Mat src;
	const int size = 60;
	if (mode == 0) {
		src = imread(getPath(srcPicNum),CV_LOAD_IMAGE_COLOR);
	}
	else {
		src = UserChosenMat;
	}
	resize(src, src, Size(size, size));
	Mat gray;
	cvtColor(src, gray, CV_RGB2GRAY);
	int tempMat[size][size];
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			int value = gray.at<uchar>(i, j);
			value /= 16;
			tempMat[i][j] = value;
			//cout << value << " ";
		}
		//cout << endl;
	}
	int dst[16][16];
	int total = 0;
	double dstNorm[16][16];
	//feature[0]:energy		能量
	//feature[1]:entropy	熵
	//feature[2]:contrast	对比度
	//feature[3]:idMomont	逆差矩
	double feature[4];
	memset(feature, 0.0, 4 * sizeof(double));
	//水平方向,从左指向右
	memset(dst, 0, 16 * 16 * sizeof(int));
	memset(dstNorm, 0.0, 16 * 16 * sizeof(double));
	total = 0;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size - 1; j++) {
			int row = tempMat[i][j];
			int col = tempMat[i][j + 1];
			dst[row][col]++;
			total++;
		}
	}//下面高斯正则化,并计算四个特征
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			//cout << dst[i][j] << " ";
			dstNorm[i][j] = (double)dst[i][j] / (double)total;
			//cout << dstNorm[i][j] << endl;
			feature[0] += dstNorm[i][j] * dstNorm[i][j];
			if(dstNorm[i][j]>0)	feature[1] -= dstNorm[i][j] * log(dstNorm[i][j]);
			feature[2] += (double)(i - j)*(double)(i - j)*dstNorm[i][j];
			feature[3] += dstNorm[i][j] / (1 + (double)(i - j)*(double)(i - j));
		}
		//cout << endl;
	}
	//竖直方向,从左指向右
	memset(dst, 0, 16 * 16 * sizeof(int));
	memset(dstNorm, 0.0, 16 * 16 * sizeof(double));
	total = 0;
	for (int i = 0; i < size - 1; i++) {
		for (int j = 0; j < size; j++) {
			int row = tempMat[i][j];
			int col = tempMat[i+1][j];
			dst[row][col]++;
			total++;
		}
	}
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			//cout << dst[i][j] << " ";
			dstNorm[i][j] = (double)dst[i][j] / (double)total;
			feature[0] += dstNorm[i][j] * dstNorm[i][j];
			if (dstNorm[i][j]>0)	feature[1] -= dstNorm[i][j] * log(dstNorm[i][j]);
			feature[2] += (double)(i - j)*(double)(i - j)*dstNorm[i][j];
			feature[3] += dstNorm[i][j] / (1 + (double)(i - j)*(double)(i - j));
		}
		//cout << endl;
	}
	//45°方向,从左上指向右下
	memset(dst, 0, 16 * 16 * sizeof(int));
	memset(dstNorm, 0.0, 16 * 16 * sizeof(double));
	total = 0;
	for (int i = 0; i < size - 1; i++) {
		for (int j = 0; j < size - 1; j++) {
			int row = tempMat[i][j];
			int col = tempMat[i + 1][j + 1];
			dst[row][col]++;
			total++;
		}
	}
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			dstNorm[i][j] = (double)dst[i][j] / (double)total;
			feature[0] += dstNorm[i][j] * dstNorm[i][j];
			if (dstNorm[i][j]>0)	feature[1] -= dstNorm[i][j] * log(dstNorm[i][j]);
			feature[2] += (double)(i - j)*(double)(i - j)*dstNorm[i][j];
			feature[3] += dstNorm[i][j] / (1 + (double)(i - j)*(double)(i - j));
		}
	}
	//135°方向,从右上指向左下
	memset(dst, 0, 16 * 16 * sizeof(int));
	memset(dstNorm, 0.0, 16 * 16 * sizeof(double));
	total = 0;
	for (int i = 0; i < size - 1; i++) {
		for (int j = 1; j < size; j++) {
			int row = tempMat[i][j];
			int col = tempMat[i + 1][j - 1];
			dst[row][col]++;
			total++;
		}
	}
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			dstNorm[i][j] = (double)dst[i][j] / (double)total;
			feature[0] += dstNorm[i][j] * dstNorm[i][j];
			if (dstNorm[i][j]>0)	feature[1] -= dstNorm[i][j] * log(dstNorm[i][j]);
			feature[2] += (double)(i - j)*(double)(i - j)*dstNorm[i][j];
			feature[3] += dstNorm[i][j] / (1 + (double)(i - j)*(double)(i - j));
		}
	}
	string result;
	for (int i = 0; i < 4; i++) {
		feature[i] /= 4;
		if (feature[i] < min_glcm_feature[i])	min_glcm_feature[i] = feature[i];
		if (feature[i] > max_glcm_feature[i])	max_glcm_feature[i] = feature[i];
		result.append(to_string(feature[i])); 
		//cout << feature[i] << endl;
		if (i != 3)	result.push_back(',');
	}
	//cout << endl << endl << endl;
	/*cout << endl << endl << result << endl;*/

	return result;
}
string Texture_global_LBP(int srcPicNum, int mode) {
	Mat src, graySrc;
	int hist[256];
	memset(hist, 0, 256 * sizeof(int));
	if (mode == 0) {
		src = imread(getPath(srcPicNum), CV_LOAD_IMAGE_COLOR);
	}
	else {
		src = UserChosenMat;
	}
	cvtColor(src, graySrc, CV_RGB2GRAY);
	resize(graySrc, graySrc, Size(graySrc.rows/4, graySrc.rows/4));
	int total = (graySrc.rows - 2)*(graySrc.cols - 2);
	int neighbor[8];//左上角，顺时针，到左边
	for (int i = 1; i < graySrc.rows-1; i++) {
		for (int j = 1; j < graySrc.cols - 1; j++) {
			memset(neighbor, 0, 8 * sizeof(int));
			uchar mid = graySrc.at<uchar>(i, j);
			neighbor[0] = (graySrc.at<uchar>(i - 1, j - 1) > mid) ? 1 : 0;
			neighbor[1] = (graySrc.at<uchar>(i - 1, j) > mid) ? 1 : 0;
			neighbor[2] = (graySrc.at<uchar>(i - 1, j + 1) > mid) ? 1 : 0;
			neighbor[3] = (graySrc.at<uchar>(i, j + 1) > mid) ? 1 : 0;
			neighbor[4] = (graySrc.at<uchar>(i + 1, j + 1) > mid) ? 1 : 0;
			neighbor[5] = (graySrc.at<uchar>(i + 1, j ) > mid) ? 1 : 0;
			neighbor[6] = (graySrc.at<uchar>(i + 1, j - 1) > mid) ? 1 : 0;
			neighbor[7] = (graySrc.at<uchar>(i, j - 1) > mid) ? 1 : 0;
			int bin = 0;
			for (int t = 0; t < 8; t++) {
				bin *= 2;
				bin += neighbor[t];
			}
			hist[bin]++;
		}
	}
	string result = to_string(total);
	result.push_back(',');
	for (int i = 0; i < 256; i++) {
		result.append(to_string(hist[i]));
		if (i != 255) result.push_back(',');
	}
	//cout << result << endl;
	return result;
}




//void Training_Color_PHash() {
//	ofstream out;
//	out.open("color_phash.txt", ios::trunc);
//	for (int i = 1; i <= MAX_PICNNUM; i++) {
//		string tmpstr = Color_PHash(i, TRAINING_MODE);
//		out << tmpstr;
//		if (i != MAX_PICNNUM)
//			out << "\n";
//	}
//	out.close();
//}

//void Training_Color_hsvhist() {
//	ofstream out;
//	out.open("color_hsvhist.txt", ios::trunc);
//	for (int i = 1; i <= MAX_PICNNUM; i++) {
//		string tmpstr = Color_hsv_hist(i, TRAINING_MODE);
//		out << tmpstr;
//		if (i != MAX_PICNNUM)	out << "\n";
//	}
//	out.close();
//}

//void Training_Color_otsu() {
//	ofstream out;
//	out.open("color_otsu.txt", ios::trunc);
//	for (int i = 1; i <= MAX_PICNNUM; i++) {
//		string tmpstr = Color_otsu(i, TRAINING_MODE);
//		out << tmpstr;
//		if (i != MAX_PICNNUM)	out << "\n";
//	}
//	out.close();
//}

//void Training_Texture_GLCM() {
//		for (int i = 0; i < 4; i++) {
//			max_glcm_feature[i] = 0;
//			min_glcm_feature[i] = 100;
//		}
//		ofstream out;
//		out.open("texture_glcm.txt", ios::trunc);
//		for (int i = 1; i <= MAX_PICNNUM; i++) {
//			string tmpstr = Texture_GLCM(i, TRAINING_MODE);
//			out << tmpstr;
//			if (i != MAX_PICNNUM)	out << "\n";
//		}
//		out.close();
//		out.open("texture_glcm_minmax.txt",ios::out);
//		string min, max;
//		for (int i = 0; i < 4; i++) {
//			min.append(to_string(min_glcm_feature[i]));
//			min.push_back(',');
//			max.append(to_string(max_glcm_feature[i]));
//			max.push_back(',');
//		}
//		out << min << " //-min" << "\n";
//		out << max << " //-max" << "\n";
//		out.close();
//}

//void Training_Texture_LBP() {
//		ofstream out;
//		out.open("texture_lbp.txt", ios::trunc);
//		for (int i = 1; i <= MAX_PICNNUM; i++) {
//			string tmpstr = Texture_global_LBP(i, TRAINING_MODE);
//			out << tmpstr;
//			if (i != MAX_PICNNUM)	out << "\n";
//		}
//		out.close();
//}

int Retrieval_Color_PHash() {
	ifstream in;
	in.open("color_phash.txt", ios::in);
	if (in.fail())   return -1;
	string phash = Color_PHash(-1, TEST_MODE);
	string line;
	int len = phash.size();
	int minRes = 1e3;
	int sign = -1;
	int res = 0;
	for (int i = 1; i <= MAX_PICNNUM; i++) {
		getline(in, line, '\n');
		//        cout << line << endl;
		res = 0;
		for (int j = 0; j<len; j++) {
			if (phash[j] != line[j]) {
				res++;
			}
		}
		if (res<minRes) {
			//            cout << minRes << endl;
			minRes = res;
			sign = i;
		}
	}
	cout << "pHash算法选出的图片是：" << sign << endl;
	cout << "差异是" << minRes << endl;
	return sign;
}
int Retrieval_Color_hsvhist_intersect() {
	ifstream in;
	in.open("color_hsvhist.txt", ios::in);
	if (in.fail())   return -1;
	string hsvcode = Color_hsv_hist(-1, TEST_MODE);
	int dstArr[64] , curArr[64];
	memset(dstArr, 0, 64 * sizeof(int));
	memset(curArr, 0, 64 * sizeof(int));
	vector<string> strvec = split(hsvcode, ",");
	for (int i = 0; i < 64; i++) {
		dstArr[i] = stoi(strvec[i]);
		//cout << dstArr[i] << endl;
	}
	string line;
	int maxRes = 0;
	int winner = -1;
	int res = 0;
	for (int i = 1; i <= MAX_PICNNUM; i++) {
		getline(in, line, '\n');
		res = 0;
		memset(curArr, 0, 64 * sizeof(int));
		strvec.clear();
		strvec = split(line, ",");
		for (int i = 0; i < 64; i++) {
			curArr[i] = stoi(strvec[i]);
			res += min(curArr[i], dstArr[i]);
		}
		if (res>maxRes) {
			cout << maxRes << endl;
			maxRes = res;
			winner = i;
		}
	}
	cout << "pHash算法选出的图片是：" << winner << endl;
	cout << "直方图上相交的像素个数是" << maxRes << endl;
	return winner;
}
int Retrieval_Color_hsvhist_Bhattacharyya() {
	ifstream in;
	in.open("color_hsvhist.txt", ios::in);
	if (in.fail())   return -1;
	string hsvcode = Color_hsv_hist(-1, TEST_MODE);
	int dstArr[64], curArr[64];
	memset(dstArr, 0, 64 * sizeof(int));
	memset(curArr, 0, 64 * sizeof(int));
	vector<string> strvec = split(hsvcode, ",");
	for (int i = 0; i < 64; i++) {
		dstArr[i] = stoi(strvec[i]);
	}
	string line;
	int maxRes = 0;
	int winner = -1;
	int res = 0;
	for (int i = 1; i <= MAX_PICNNUM; i++) {
		getline(in, line, '\n');
		res = 0;
		memset(curArr, 0, 64 * sizeof(int));
		strvec.clear();
		strvec = split(line, ",");
		for (int i = 0; i < 64; i++) {
			curArr[i] = stoi(strvec[i]);
			res += (int)(sqrt(curArr[i]* dstArr[i]));
		}
		if (res>maxRes) {
			cout << maxRes << endl;
			maxRes = res;
			winner = i;
		}
	}
	cout << "pHash算法选出的图片是：" << winner << endl;
	cout << "直方图上相交的像素乘积个数是" << maxRes << endl;
	return winner;
}
int Retrieval_Color_otsu() {
	ifstream in;
	in.open("color_otsu.txt", ios::in);
	if (in.fail())   return -1;
	string otsu = Color_otsu(-1, TEST_MODE);
	string line;
	int len = otsu.size();
	int minRes = 1e4;
	int winner = -1;
	int res = 0;
	for (int i = 1; i <= MAX_PICNNUM; i++) {
		getline(in, line, '\n');
		//        cout << line << endl;
		res = 0;
		for (int j = 0; j<len; j++) {
			if (otsu[j] != line[j]) {
				res++;
			}
			if (res >= minRes) break;
		}
		if (res<minRes) {
			//            cout << minRes << endl;
			minRes = res;
			winner = i;
		}
	}
	cout << "otsu算法选出的图片是：" << winner << endl;
	cout << "差异是" << minRes << endl;
	return winner;
}
int Retrieval_Texture_glcm() {
	ifstream in;
	string line;
	double min[4], range[4];
	in.open("texture_glcm_minmax.txt", ios::in);
	getline(in, line, '\n');
	vector<string> minstr = split(line, ",");
	getline(in, line, '\n');
	vector<string> maxstr = split(line, ",");
	for (int i = 0; i < 4; i++) {
		//cout << minstr[i] << endl;
		min[i] = stod(minstr[i]);
		range[i] = stod(maxstr[i]) - min[i];
	}
	in.close();
	in.open("texture_glcm.txt", ios::in);
	if (in.fail())   return -1;
	string gclm_feature = Texture_GLCM(-1, TEST_MODE);
	vector<string> strvec = split(gclm_feature, ",");
	double vec[4];
	double alen = 0;
	for (int i = 0; i < 4; i++) {
		vec[i] = (stod(strvec[i]) - min[i]) / range[i];
		alen += vec[i] * vec[i];
	}
	alen = sqrt(alen);
	double maxRes = -2;
	int winner = -1;
	double cosRes = 0;
	double curvec[4];
	double blen = 0;
	double product = 0;
	for (int i = 1; i <= MAX_PICNNUM; i++) {
		if (i % 100 == 0)	cout << i << " 处理完成"<< endl;
		getline(in, line, '\n');
		cosRes = 0;
		blen = 0;
		product = 0;
		strvec.clear();
		strvec = split(line, ",");
		for (int i = 0; i < 4; i++) {
			curvec[i] = (stod(strvec[i]) - min[i]) / range[i];
			product += curvec[i] * vec[i];
			blen += curvec[i] * curvec[i];
		}
		blen = sqrt(blen);
		cosRes = product / (alen*blen);
		//cout << i << "图片： "<< cosRes << endl;
		if (cosRes  > maxRes) {
			cout << setprecision(14) << maxRes << endl;
			maxRes = cosRes;
			winner = i;
		}
	}
	cout << "glcm算法选出的图片是：" << winner << endl;
	cout << setprecision(14) << "余弦相似度是" << maxRes << endl;
	return winner;

}
int Retrieval_Texture_lbp() {
	ifstream in;
	in.open("texture_lbp.txt", ios::in);
	if (in.fail())   return -1;
	string lbp = Texture_global_LBP(-1, TEST_MODE);
	double dstArr[256];
	vector<string> strvec = split(lbp, ",");
	int total = stoi(strvec[0]);
	for (int i = 0; i < 255; i++) {
		dstArr[i] = (double)(stoi(strvec[i + 1]))/(double)total;
	}
	string line;
	double maxRes = 0;
	int winner = -1;
	double res = 0;
	double curbin;
	for (int i = 1; i <= MAX_PICNNUM; i++) {
		getline(in, line, '\n');
		res = 0;
		strvec.clear();
		strvec = split(line, ",");
		total = stoi(strvec[0]);
		for (int i = 0; i < 255; i++) {
			curbin = (double)(stoi(strvec[i + 1])) / (double)total;
			res += min(curbin, dstArr[i]);
		}
		if (res>maxRes) {
			cout << maxRes << endl;
			maxRes = res;
			winner = i;
		}
		if (i % 200 == 0) cout << "处理完毕" << i << endl;
	}
	cout << "lbp算法选出的图片是：" << winner << endl;
	cout << "结果数值是" << maxRes << endl;
	return winner;
}

int main(int argc, char *argv[])
{
	DWORD starttime = GetTickCount();
	//Training_Color_PHash();
	//Training_Color_hsvhist();
	//Training_Color_otsu();
	//Training_Texture_GLCM();
	//Training_Texture_LBP();

	string chosenPath = "E:\\programming\\MATLAB\\pictures\\Einstein01.jpg";
	UserChosenMat = imread(chosenPath,CV_LOAD_IMAGE_COLOR);
	namedWindow("src",CV_WINDOW_AUTOSIZE);
	imshow("src",UserChosenMat);

	int res = Retrieval_Texture_lbp();

	showPic(res,"result");

	DWORD endtime = GetTickCount();
	long elapseTime = (endtime - starttime) / 1000;
	cout << "用时" << elapseTime << "秒" << endl;

	waitKey(0);
	system("pause");
	return 0;
}


