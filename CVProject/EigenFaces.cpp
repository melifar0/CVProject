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
Mat _testingData;
//gets eigenvectors for training images
void EigenFaces::Eigenfaces(const vector<Mat> imgSet) {
	//check for invalid input
	
	if (imgSet.size() == 0) {
		printf("Empty image set provided");
		return;
	}

	//transform input to a row vector per image 
	//number of images is number of rows of new matrix
	int height = imgSet.size();
	//total number of pixels is the size of the columns of new matrix
	int width = imgSet[0].total();
	//printf("heigth and width are %d, %d \n", height, width);
	_data = Mat(height, width, imgSet[0].type());
	_data = getDataAsRows(imgSet);

	if (_data.type() != CV_32FC1 || _data.type() != CV_64FC1 || _data.type() != CV_32FC2 || _data.type() != CV_64FC2) {
		cout << "format is =  \n" << _data.type() << '\n';
		_data.convertTo(_data, CV_32FC1);
		printf("Matrix does not have right format\n");
		cout << "format is = " << _data.type() << '\n';
	}
	Mat cov = Mat(height, height, CV_32F);
	_mean = Mat(1, _data.cols, CV_32F, 0);
	Mat tempEigen;
	calcCovarMatrix(_data, cov, _mean, CV_COVAR_SCRAMBLED + CV_COVAR_ROWS + CV_COVAR_SCALE, CV_32F);

	bool ret;
	cout << "I get covariance matrix \n";
	
	eigen(cov, _eigenvalues, tempEigen);

	cout << "I get temp eigenvalues \n";

	//minus mean from data for eigenvector calculation purpose
	Mat temp_data = _data.clone();
	for (int i = 0; i < temp_data.rows; i++) {
		subtract(temp_data.row(i), _mean, temp_data.row(i));
	}
	
	_eigenvectors = Mat(height, width, CV_32F);
	gemm(tempEigen, temp_data, 1, Mat(), 0, _eigenvectors, 0);

	cout << "I multiply eigenvalues \n";

	//normalize
	for (int i = 0; i < height; i++) {
		normalize(_eigenvectors.row(i), _eigenvectors.row(i));
	}

	cout << "I get eigenvectors\n";

	transpose(_eigenvectors, _eigenvectors);


}

void EigenFaces::crossValidation(ImageDataSet data) {


	vector<Mat> trainingSet;
	vector<Mat> testSet;
	vector<string> trainingSetLabels;
	vector<string> testSetLabels;
	vector<double> reconstructionErrorsTraining;
	vector<double> reconstructionErrorsTesting;
	
	for (int i = 0; i < 13; i++) {
		reconstructionErrorsTraining.push_back(0);
	}

	for (int i = 0; i < 13; i++) {
		reconstructionErrorsTesting.push_back(0);
	}
	/*
	//ACTUAL K-Fold cross validation
	KFold kfold;
	vector<vector<Mat>> partitioned = kfold.getPartitionedData(data);
	vector<vector<string>> labels = kfold.getLabels();

	//get the test set and training set (different each time)
	for (int kOfTest = 0; kOfTest < 3; kOfTest++){
		for (int i = 0; i < 3; i++){
			//insert kOfTest partition as test set
			if (i == kOfTest) {
				testSet.insert(testSet.end(), partitioned[i].begin(), partitioned[i].end());
				testSetLabels.insert(testSetLabels.end(), labels[i].begin(), labels[i].end());
			}
			//else its part of the training set
			else {
				trainingSet.insert(trainingSet.end(), partitioned[i].begin(), partitioned[i].end());
				trainingSetLabels.insert(trainingSetLabels.end(), labels[i].begin(), labels[i].end());
			}
		}

		//now we have training and test set. Call EigenFaces on training set in order to produce eigenvectors and eigenvalues
		cout << "size of training set =" << trainingSet.size() << '\n';
		cout << "size of test set =" << testSet.size() << '\n';
		//convert training set and test set to Mat with each row being one image
		//_testingData = getDataAsRows(testSet);
		//get eigenvectors for trainingData
		Eigenfaces(trainingSet);

		//this is the eigenvectors we need to perform projection and reconstruction with
		vector<int> eigenvectorsTest = { 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, _eigenvectors.cols };

		for (int eigenvectorIdx = 0; eigenvectorIdx < eigenvectorsTest.size(); eigenvectorIdx++){
			int numEigenvectors = eigenvectorsTest[eigenvectorIdx];

			//calculate the projection for training data
			Mat projection = project(numEigenvectors, _data);

			//get the reconstuction matrix
			Mat reconstruction;
			//calculate reconstruction for training data
			reconstruction = reconstruct(numEigenvectors, projection);

			//get the difference between the reconstruction and the image
			Mat differenceMatrix;
			//absolute difference 
			absdiff(reconstruction, _data, differenceMatrix);
			double reconstructionError = sum(differenceMatrix)[0];
			//add reconstruction error to array of errors for plotting
			reconstructionErrorsTraining[eigenvectorIdx] += reconstructionError;
			cout << "reconstruction error training is " << reconstructionError << '\n';

			//get testingData
			_testingData = getDataAsRows(testSet);
			_testingData.convertTo(_testingData, CV_32F);

			//repeat process for testing data
			Mat projectionTesting = project(numEigenvectors, _testingData);

			//get the reconstuction matrix
			Mat reconstructionTesting;
			//calculate reconstruction for training data
			reconstructionTesting = reconstruct(numEigenvectors, projectionTesting);

			//get the difference between the reconstruction and the image
			Mat differenceMatrixTesting;
			//absolute difference 
			absdiff(reconstructionTesting, _testingData, differenceMatrixTesting);
			double reconstructionErrorTesting = sum(differenceMatrixTesting)[0];
			//add reconstruction error to array of errors for plotting
			reconstructionErrorsTesting[eigenvectorIdx] += reconstructionErrorTesting;
			cout << "reconstruction error testing is " << reconstructionErrorTesting << '\n';

			int numCorrectlyPredicted = 0;
			//make prediction
			for (int testingIdx = 0; testingIdx < projectionTesting.rows; testingIdx++) {
				int predictedIndex = makePrediction(projection, projectionTesting.row(testingIdx));
				//compare the predictions
				if (predictedIndex >= 0) {
					if (testSetLabels[testingIdx].compare(trainingSetLabels[predictedIndex]) == 0){
						numCorrectlyPredicted++;
					}
				}
			}
			cout << "percentage of correctly predicted images = " << (double)numCorrectlyPredicted*100.0 / testSet.size() << "\n";

			//display original and reconstructed image
			//display the reconstruction image
			double min, max;
			minMaxLoc(reconstruction, &min, &max);
			cout << "min for reconstruction is " << min << '\n';
			cout << "max for reconstruction is" << max << '\n';
			Mat reconstructionToDisplay;
			reconstruction.convertTo(reconstructionToDisplay, CV_8U);
			imshow("reconstruction image", reconstruction.row(0).reshape(0, 100));
			waitKey(0);

			//display the original image
			minMaxLoc(_data, &min, &max);
			cout << "min for _data is " << min << '\n';
			cout << "max for _data is" << max << '\n';
			Mat dataToDisplay;
			_data.convertTo(dataToDisplay, CV_8U);
			imshow("data image", _data.row(0).reshape(0, 100));
			waitKey(0);


		}

		//clear training set and test set which will be recreated for next fold
		testSet.clear();
		trainingSet.clear();
	}

	//now that we got the reconstruction error just show it
	for (int i = 0; i < 13; i++) {
		cout << "Reconstruction error " << i << " is " << reconstructionErrorsTraining[i]/7<< "\n";
	}
	*/
	
	//temporary for testing purposes
	
	for (int i = 0; i < data.QMUL_SubjectIDs.size(); i++) {

		//get only one pose for testing
		Mat grey = Mat(100, 100, CV_32FC1);
		cvtColor(data.QMUL_getSubjectImageByPose(data.QMUL_SubjectIDs[i], "090", "090"), grey, CV_BGR2GRAY);
		trainingSet.push_back(grey);
	}

	
	//call the function to produce eigenvectors
	Eigenfaces(trainingSet);


	//testing purposes with 50 eigenvectors
	int numEigenvectors = 31;

	//calculate the projection
	Mat projection = project(numEigenvectors, _data);

	//get the reconstuction matrix
	Mat reconstruction;

	reconstruction = reconstruct(numEigenvectors, projection);

	//get the reconstruction error
	Mat differenceMatrix;
	absdiff(reconstruction, _data, differenceMatrix);
	double reconstructionError = sum(differenceMatrix)[0];

	cout << "reconstruction error is " << reconstructionError/(31*10000) << '\n';

	//get reconstruction for one image at a time
	for (int i = 0; i < 31; i++){
		double temp_error = 0;
		double temp_error2 = 0;
		Mat error = Mat(1, 10000, CV_32F, 0);
		error.setTo(0);
		absdiff(_data.row(i), reconstruction.row(i), error);
		temp_error = sum(error)[0];
		//cout << error.reshape(0, 100).row(0);
		cout << "reconstruction error is "<< temp_error << "\n";
		
	}

	
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
	minMaxLoc(reconstruction.row(1), &min, &max);
	cout << "min for reconstruction is " << min << '\n';
	cout << "max for reconstruction is" << max << '\n';
	reconstruction.convertTo(reconstruction, CV_8U, 255.0/(max-min));
	imshow("reconstruction image", reconstruction.row(1).reshape(0, 100));
	waitKey(0);

	//display the original image
	minMaxLoc(_data.row(1), &min, &max);
	cout << "min for _data is " << min << '\n';
	cout << "max for _data is" << max << '\n';
	_data.convertTo(_data, CV_8U);
	imshow("data image", _data.row(1).reshape(0, 100));
	waitKey(0);

	//display the projection
	cout << "projection dimensions = " << projection.rows << "x" << projection.cols << "\n";
	minMaxLoc(projection, &min, &max);
	cout << "min for projection is " << min << '\n';
	cout << "max for projection is" << max << '\n';
	//projection.convertTo(projection, CV_8U, 255.0/(max-min));
	//imshow("projection image", projection.row(0).reshape(0, 100));
	//waitKey(0);
	
	
}

Mat EigenFaces::project(int numEigenvectors, Mat input){
	Mat subsetOfEigenvectors;
	//get current subset of eigenvectors 

	//now eigenvectors stored by column so get numEienvectors columns TODO: CHECK VALIDITY 
	_eigenvectors(Rect(0, 0, numEigenvectors, _eigenvectors.rows)).copyTo(subsetOfEigenvectors);

	//ensure all data types are the same
	if (input.type() != CV_32F || _mean.type() != CV_32F || subsetOfEigenvectors.type() != CV_32F) {
		input.convertTo(input, CV_32F);
		_mean.convertTo(_mean, CV_32F);
		subsetOfEigenvectors.convertTo(subsetOfEigenvectors, CV_32F);
		cout << "i do a conversion here \n";
	}
	//subtract the mean from the data
	for (int i = 0; i < input.rows; i++) {
		subtract(input.row(i), _mean, input.row(i));
	}


	//get the projection matrix
	Mat projection = input*subsetOfEigenvectors;

	cout << "I get projection\n";

	//add the mean matrix again to the data
	for (int i = 0; i < input.rows; i++) {
		add(input.row(i), _mean, input.row(i));
	}


	return projection;
}

int EigenFaces::makePrediction(Mat trainingProjections, Mat testingProjection){

	double minimumDistance = DBL_MAX;
	int predictedRow = -1;

	for (int i = 0; i < trainingProjections.rows; i++) {
		double distance = norm(trainingProjections.row(i), testingProjection, NORM_L2);
		if (distance < minimumDistance) {
			minimumDistance = distance;
			predictedRow = i;
		}
	}
	return predictedRow;
}
Mat EigenFaces::reconstruct(int numEigenvectors, Mat projection){
	Mat subsetOfEigenvectors;
	//now eigenvectors stored by column so get numEienvectors columns TODO: CHECK VALIDITY 
	_eigenvectors(Rect(0, 0, numEigenvectors, _eigenvectors.rows)).copyTo(subsetOfEigenvectors);

	//ensure all data types are the same
	if (projection.type() != CV_32F || _mean.type() != CV_32F || subsetOfEigenvectors.type() != CV_32F) {
		projection.convertTo(projection, CV_32F);
		_mean.convertTo(_mean, CV_32F);
		subsetOfEigenvectors.convertTo(subsetOfEigenvectors, CV_32F);
		cout << "I make a convesion in reconstruction\n";
	}
	//transpose subset of eigenvectors
	Mat reconstruction;
	gemm(projection, subsetOfEigenvectors, 1.0, Mat(), 0.0, reconstruction, GEMM_2_T);
	cout << "I get reconstruction\n";
	//add the mean matrix to the reconstruction
	double min, max;
	minMaxLoc(projection, &min, &max);
	cout << "min for reco is " << min << '\n';
	cout << "max for reco is" << max << '\n';

	//imshow("reco", reconstruction.row(0).reshape(100));
	//waitKey(0);

	for (int i = 0; i < reconstruction.rows; i++) {
		add(reconstruction.row(i), _mean, reconstruction.row(i));
	}

	return reconstruction;

}

Mat EigenFaces::getDataAsRows(vector<Mat> src){
	//transform input to a row vector per image 
	//number of images is number of rows of new matrix
	int height = src.size();
	//total number of pixels is the size of the columns of new matrix
	int width = src[0].total();
	printf("heigth and width are %d, %d \n", height, width);
	Mat ret = Mat(height, width, src[0].type());
	for (int i = 0; i < height; i++) {
		if (src[i].isContinuous()){
			src[i].reshape(0, 1).convertTo(ret.row(i), ret.type(), 1, 0);
		}
		else {
			src[i].clone().reshape(0, 1).copyTo(ret.row(i));
		}
	}

	return ret;
}