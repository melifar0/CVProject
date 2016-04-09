#ifndef BOWFACES_H
#define BOWFACES_H

#include <opencv2/opencv.hpp>
#include <string>
#include "ImageDataSet.h"


using namespace cv;
using namespace std;

class BOWFaces {
public:
	void BOWcrossValidation(ImageDataset data);
	void BOWTrain(const vector<Mat>& imgSet, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords)
	void BOWTest(const vector<Mat>& imgSet, const Mat &codeBook, const vector<vector<Mat>> &imageDescriptors)
};
#endif