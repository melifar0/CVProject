#ifndef EIGENFACES_H
#define EIGENFACES_H
#endif

#include <opencv2/opencv.hpp>
#include <string>
#include "ImageDataSet.h"


using namespace cv;
using namespace std;

class EigenFaces {
public:
	void Eigenfaces(const vector<Mat> imgSet);
	void crossValidation(ImageDataSet data);
	Mat project(int numEigenvectors, Mat input);
	Mat reconstruct(int numEigenvectors, Mat projection);
	Mat getDataAsRows(vector<Mat> src);
};
