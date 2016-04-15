#ifndef BOWFACES_H
#define BOWFACES_H

#include <opencv2/opencv.hpp>
#include <string>
#include "ImageDataSet.h"


using namespace cv;
using namespace std;

class BOWFaces {
public:
	void BOWcrossValidation(ImageDataSet data);
	void BOWTrain(const vector<Mat>& imgSet, Mat &codeBook, vector<Mat> &BOWhistrow);
	void BOWcrossValidationProb(ImageDataSet data);
	void BOWTrainProb(const vector<Mat>& imgSet, Mat &codeBook, vector<Mat> &BOWhistrow, const vector<string> &trainingSetLabels);
	double chiSquareDist(InputArray _H1, InputArray _H2);
private:
	double BOWTest(const vector<Mat>& imgSet, const Mat &codeBook, const vector<Mat> &BOWhistrow, const vector<string> &trainingSetLabels, const vector<string> &testSetLabels);
	double BOWTestProb(const vector<Mat>& imgSet, const Mat &codeBook, const vector<Mat> &BOWhistrow, const vector<string> &trainingSetLabels, const vector<string> &testSetLabels);
};
#endif