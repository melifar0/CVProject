#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "ImageDataSet.h"
#include "kFold.h"
#include "ImageDataSet.h"
#include "EigenFaces.h"


using namespace cv;
using namespace std;

Mat _eigenvectors;
Mat _eigenvalues;
Mat _mean;
Mat _data;
void EigenFaces::Eigenfaces(const vector<Mat>& imgSet) {
	//check for invalid input
	if (imgSet.size() == 0) {
		printf("Empty image set provided");
		return;
	}

	cout << "format is = " << imgSet[0].type() << '\n';

	//transform input to a row vector per image 
	//number of images is number of rows of new matrix
	int height = imgSet.size();
	//total number of pixels is the size of the columns of new matrix
	int width = imgSet[0].total();
	printf("heigth and width are %d, %d \n", height, width);
	_data = Mat(height, width, imgSet[0].type());
	//create data matrix
	for (int i = 0; i < height; i++) {
		if (imgSet[i].isContinuous()){
			imgSet[i].reshape(0, 1).convertTo(_data.row(i), _data.type(), 1, 0);
		}
		else {
			imgSet[i].clone().reshape(0, 1).copyTo(_data.row(i));
		}
	}


	if (_data.type() != CV_32FC1 || _data.type() != CV_64FC1 || _data.type() != CV_32FC2 || _data.type() != CV_64FC2) {
		cout << "format is =  \n" << _data.type() << '\n';
		_data.convertTo(_data, CV_32FC1);
		printf("Matrix does not have right format\n");
		cout << "format is = " << _data.type() << '\n';
		cout << "format is = " << CV_32FC1 << '\n';
	}
	Mat cov;
	_mean = Mat(1, _data.cols, CV_32F, 0);
	calcCovarMatrix(_data, cov, _mean, CV_COVAR_NORMAL + CV_COVAR_ROWS);

	bool ret;
	cout << "I get covariance matrix \n";
	ret = eigen(cov, _eigenvalues, _eigenvectors);

}

void EigenFaces::crossValidation(ImageDataSet data) {





	vector<Mat> trainingSet;
	vector<Mat> testSet;
	/*
	ACTUAL K-Fold cross validation
	KFold kfold;
	vector<vector<Mat>> partitioned = kfold.getPartitionedData(data);
	testSet.insert(testSet.end(), partitioned[0].begin(), partitioned[0].end());
	for (int i = 1; i < 7; i++){
	trainingSet.insert(trainingSet.end(), partitioned[i].begin(), partitioned[i].end());
	}
	*/
	//temporary for testing purposes
	for (int i = 0; i < data.QMUL_SubjectIDs.size(); i++) {

		//get only one pose for testing
		Mat grey = Mat(100, 100, CV_32FC1);
		cvtColor(data.QMUL_getSubjectImageByPose(data.QMUL_SubjectIDs[i], "090", "090"), grey, CV_BGR2GRAY);
		trainingSet.push_back(grey);
	}

	cout << "size of training set =" << trainingSet.size() << '\n';
	cout << "size of test set =" << testSet.size() << '\n';


	//call the function to produce eigenvectors
	Eigenfaces(trainingSet);

	vector<int> eigenvectorsTest = { 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, _eigenvalues.rows };

	//testing purposes with 50 eigenvectors
	int numEigenvectors = 1000;

	Mat subsetOfEigenvectors;
	//get current subset of eigenvectors 
	//eigenvectors are stored by row so need to transpose
	transpose(_eigenvectors, _eigenvectors);
	//now eigenvectors stored by column so get numEienvectors columns
	_eigenvectors(Rect(0, 0, numEigenvectors, _eigenvectors.rows)).copyTo(subsetOfEigenvectors);

	//ensure all data types are the same
	if (_data.type() != CV_32F || _mean.type() != CV_32F || subsetOfEigenvectors.type() != CV_32F) {
		_data.convertTo(_data, CV_32F);
		_mean.convertTo(_mean, CV_32F);
		subsetOfEigenvectors.convertTo(subsetOfEigenvectors, CV_32F);
	}
	//subtract the mean from the data
	for (int i = 0; i < _data.rows; i++) {
		subtract(_data.row(i), _mean, _data.row(i));
	}


	//get the projection matrix
	Mat projection = _data * subsetOfEigenvectors;


	//try adding the mean to the projection
	for (int i = 0; i < projection.rows; i++) {
		add(projection.row(i), _mean, projection.row(i));
	}

	//get the reconstuction matrix
	Mat reconstruction;
	gemm(projection, subsetOfEigenvectors, 1.0, Mat(), 0.0, reconstruction, GEMM_2_T);
	//add the mean matrix to the reconstruction
	/*
	for (int i = 0; i < reconstruction.rows; i++) {
		add(reconstruction.row(i), _mean, reconstruction.row(i));
	}*/
	//add the mean matrix again to the data for comparison purposes
	for (int i = 0; i < _data.rows; i++) {
		add(_data.row(i), _mean, _data.row(i));
	}

	//get the reconstruction error
	Mat differenceMatrix;
	subtract(reconstruction, _data, differenceMatrix);
	double reconstructionError = sum(differenceMatrix)[0];

	cout << "reconstruction error is " << reconstructionError << '\n';

	//display the average image
	//convert first
	//find max and min
	double min, max;
	minMaxLoc(_mean, &min, &max);
	cout << "min is " << min << '\n';
	cout << "max is" << max << '\n';
	_mean.convertTo(_mean, CV_8U);
	imshow("mean image", _mean.reshape(0, 100));
	waitKey(0);

	//display the reconstruction image
	minMaxLoc(reconstruction, &min, &max);
	cout << "min for reconstruction is " << min << '\n';
	cout << "max for reconstruction is" << max << '\n';
	reconstruction.convertTo(reconstruction, CV_8U);
	imshow("reconstruction image", reconstruction.row(0).reshape(0, 100));
	waitKey(0);

	//display the original image
	minMaxLoc(_data, &min, &max);
	cout << "min for _data is " << min << '\n';
	cout << "max for _data is" << max << '\n';
	_data.convertTo(_data, CV_8U);
	imshow("data image", _data.row(0).reshape(0, 100));
	waitKey(0);

}