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

const vector<string> subjectNames = vector<string>
({
	"AdamB", "AndreeaV", "CarlaB", "ColinP", "DanJ",
	"DennisP", "DennisPNoGlasses", "DerekC", "GrahamW", "HeatherL",
	"Jack", "JamieS", "JeffNG", "John", "OngEJ",
	"KateS", "KatherineW", "KeithC", "KrystynaN", "PaulV",
	"RichardB", "RichardH", "SarahL", "SeanG", "SeanGNoGlasses"
	, "SimonB", "SueW", "TasosH", "TomK", "YogeshR", "YongminY"
});

Mat _eigenvectors;
Mat _eigenvalues;
Mat _mean;
Mat _data;
Mat _testingData;
PCA _pca;
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
	_data = Mat(height, width, CV_32F);
	_data = getDataAsRows(imgSet);

	if (_data.type() != CV_32FC1 || _data.type() != CV_64FC1 || _data.type() != CV_32FC2 || _data.type() != CV_64FC2) {
		cout << "format is =  \n" << _data.type() << '\n';
		_data.convertTo(_data, CV_32FC1);
		printf("Matrix does not have right format\n");
	}
	Mat cov = Mat(height, height, CV_32F);
	_mean = Mat(1, _data.cols, CV_32F, 0);
	
	Mat tempEigen;
	calcCovarMatrix(_data, cov, _mean, CV_COVAR_SCRAMBLED + CV_COVAR_ROWS + CV_COVAR_SCALE, CV_32F);

	bool ret;
	
	eigen(cov, _eigenvalues, tempEigen);

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

	for (int i = 0; i < height; i++) {
		cout << "norm is " << norm(_eigenvectors.row(i)) << "\n";
	}


	transpose(_eigenvectors, _eigenvectors);
	

	
}
vector<Mat> _meansOfSubjects; //store the mean of each subject
vector<Mat> _inverseCovarianceOfSubjects; //store the inverse covariance of each subject
vector<Mat> _covarianceOfSubjects;
//finds the probabilistic parameters for each subject ie covariance and mean and inverts covariance
void EigenFaces::probEigenfaces(vector<Mat> trainingImages, vector<string> labels){

	//find the Eigenspace of all training images using optimal number of eigenvectors
	Eigenfaces(trainingImages);
	cout << "I get after eigenvalues\n";
	Mat projection = project(100, _data); 
	//Mat projection = Mat(trainingImages.size(), 100, CV_32F);
	//_pca.project(_data).convertTo(projection, CV_32F);
	cout << "projections are " << projection.rows << "x" << projection.cols << "\n";
	cout << "I get after projection\n";
	//_meansOfSubjects.resize(subjectNames.size()); //store the mean of each subject
	//_inverseCovarianceOfSubjects.resize(subjectNames.size());//store the inverse covariance of each subject
	vector<Mat> projectionsOfSubjects;
	
	projectionsOfSubjects.resize(subjectNames.size());
	//go through all labels and separate projection per subject (inefficient but has to be done due to code structure)
	// and store their projections in projectionsOfSubjects
	for (int i = 0; i < subjectNames.size(); i++){
		//add projection for all projections of the subject
		for (int j = 0; j < labels.size(); j++) {
			if (labels[j].compare(subjectNames[i]) == 0){
				//add the projection to the projectionsOfsubjects as a row in a matrix
				projectionsOfSubjects[i].push_back(projection.row(j)); //temporary
			}
		}

	}
	//now we have the projections separated by subject so find the mean and the covariance of each subject
	for (int i = 0; i < subjectNames.size(); i++){
		//covariance is 100x100 matrix since 100 eigenvectors are used
		Mat covariance = Mat(100, 100, CV_32F);
		Mat mean = Mat(1, 100, CV_32F,0);
		calcCovarMatrix(projectionsOfSubjects[i], covariance, mean, CV_COVAR_ROWS + CV_COVAR_NORMAL + CV_COVAR_SCALE, CV_32F);
		//cout << covariance;
		
		//create a 100x100 identity matrix in order to remove non-diagonal entries of the covariance matrix
		Mat ident = Mat::eye(100, 100, CV_32F);
		covariance = covariance.mul(ident);
		//double det = determinant(covariance);
		//cout << "cov det is " << det << "\n";
		_covarianceOfSubjects.push_back(covariance);
		int nonZero = countNonZero(covariance);
		//cout << "non zero is " << nonZero << "\n";

		//cout << covariance;
		//now that all off diagonal entries are removed find the inverse of covariance
		Mat inverseCov = covariance.inv();
		//add inverse of covariance to vector of covariances
		_inverseCovarianceOfSubjects.push_back(inverseCov);
		//add the mean to vector of means
		_meansOfSubjects.push_back(mean);
	}

	//scale the diagonal of covaariance by dividing by max in the set
	double max = DBL_MIN;
	double min = DBL_MAX;
	for (int i = 0; i < _covarianceOfSubjects.size(); i++) {
		double temp_max;
		double temp_min;
		minMaxLoc(_covarianceOfSubjects[i], &temp_min, &temp_max);
		if (temp_max > max) {
			max = temp_max;
		}
		if (temp_min < min) {
			min = temp_min;
		}
	}
	cout << "max = " << max<< "\n";
	cout << "min = " << min << "\n";

	for (int i = 0; i < _covarianceOfSubjects.size(); i++) {
		_covarianceOfSubjects[i] = _covarianceOfSubjects[i] * 100/max;
	}
}
/*
With a testing projection as input, it uses the inverse covariance, covariance and mean of the 
training set in order to fit a gaussian to the testing set and calculate the recognition error
*/
string EigenFaces::probRecognition(Mat testProjection){

	//iterate through all the the gaussian and get the one with the highest likelihood
	double maxLikelihood = -1; //likelihoods cant be negative
	string idOfMaxLikelihood = "";
	for (int i = 0; i < subjectNames.size(); i++){
		//get the likelihood given the covariance and mean
		double det = determinant(_covarianceOfSubjects[i]);
		Mat diff;
		subtract(testProjection, _meansOfSubjects[i], diff);
		Mat diff_transpose;
		transpose(diff, diff_transpose);
		Mat restOfExponent = -0.5*diff * _inverseCovarianceOfSubjects[i] * diff_transpose;
		//cout << restOfExponent << "\n";
		//cout << "this should be a 1x1 matrix "<<restOfExponent.rows << "X" << restOfExponent.cols << "\n";
		double restOfExponentInDouble = sum(restOfExponent)[0];
		//cout << "determinant is " << det << "\n";
		//cout << "restOfExponentInDouble is" << restOfExponentInDouble << "\n";

		double likelihood = pow(det, -0.5)*pow(2.71, restOfExponentInDouble);
		//cout << "likelihood isssss = "<< restOfExponentInDouble << "\n";
		if (likelihood > maxLikelihood) {
			maxLikelihood = likelihood;
			idOfMaxLikelihood = subjectNames[i];
		}
	}
	//cout << "Max likelihood is " << maxLikelihood << "\n";
	return idOfMaxLikelihood;
}
/*
Does 7 fold cross validation for the probabilistic part
*/
void EigenFaces::crossValidationProb(ImageDataSet data){
	vector<Mat> trainingSet;
	vector<Mat> testSet;
	vector<string> trainingSetLabels;
	vector<string> testSetLabels;
	double recognitionRateTesting = 0;
	int numCorrectlyPredicted = 0;

	KFold kfold;
	vector<vector<Mat>> partitioned = kfold.getPartitionedData(data);
	vector<vector<string>> labels = kfold.getLabels();
	int sizeOfTest = partitioned[0].size();

	//get the test set and training set (different each time)
	for (int kOfTest = 0; kOfTest < 7; kOfTest++){
		for (int i = 0; i < 7; i++){
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

		//now we have training and test set. Call prob on trainingSet in order to get the gaussian parameters
		probEigenfaces(trainingSet, trainingSetLabels);
		//now we have gaussian parameters, make a projection of the test set to get the test vectors
		Mat testingData = getDataAsRows(testSet);
		Mat projectionOfTestSet = project(100, testingData);
		//now that we have projection of testSet find max probability given that we have the elements needed
		for (int i = 0; i < projectionOfTestSet.rows; i++){
			string actualLabel = testSetLabels[i];
			string predictedLabel = probRecognition(projectionOfTestSet.row(i));
			if (actualLabel.compare(predictedLabel) == 0){
				numCorrectlyPredicted++;
			}
		}

		trainingSet.clear();
		testSet.clear();
		trainingSetLabels.clear();
		testSetLabels.clear();
		_meansOfSubjects.clear();
		_inverseCovarianceOfSubjects.clear();
		_covarianceOfSubjects.clear();

	}
	recognitionRateTesting = (double)numCorrectlyPredicted * 100 / (7 * (double)sizeOfTest);
	cout << "recognition rate is " << recognitionRateTesting << "\n";
}
/*
Takes as input a data set. 
Does 7 fold cross validation for reconstruction error and recognition rate
*/
void EigenFaces::crossValidation(ImageDataSet data) {


	vector<Mat> trainingSet;
	vector<Mat> testSet;
	vector<string> trainingSetLabels;
	vector<string> testSetLabels;
	vector<double> reconstructionErrorsTraining;
	vector<double> reconstructionErrorsTesting;
	vector<double> recognitionRateTesting;

	for (int i = 0; i < 13; i++) {
		reconstructionErrorsTraining.push_back(0);
	}

	for (int i = 0; i < 13; i++) {
		reconstructionErrorsTesting.push_back(0);
	}

	for (int i = 0; i < 13; i++) {
		recognitionRateTesting.push_back(0);
	}
	
	//ACTUAL K-Fold cross validation
	
	KFold kfold;
	vector<vector<Mat>> partitioned = kfold.getPartitionedData(data);
	vector<vector<string>> labels = kfold.getLabels();

	//get the test set and training set (different each time)
	for (int kOfTest = 0; kOfTest < 7; kOfTest++){
		for (int i = 0; i < 7; i++){
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
		int numTimestoDisp = 1;
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
			//cout << "reconstruction error training is " << reconstructionError << '\n';

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
			int numTimesToDisplayCor = 2;
			int numTimesToDisplayIncorr = 2;
			//make prediction
			for (int testingIdx = 0; testingIdx < projectionTesting.rows; testingIdx++) {
				int predictedIndex = makePrediction(projection, projectionTesting.row(testingIdx));
				//compare the predictions
				if (predictedIndex >= 0) {
					if (testSetLabels[testingIdx].compare(trainingSetLabels[predictedIndex]) == 0){
						numCorrectlyPredicted++;

						if (numEigenvectors == 100 && numTimesToDisplayCor >= 0 && kOfTest == 0){
							//display training image for correctly classified
							Mat trainToDisplay;
							_data.row(predictedIndex).convertTo(trainToDisplay, CV_8U);
							namedWindow("correct pred train", WINDOW_AUTOSIZE);
							imshow("correctly classified test", trainToDisplay.reshape(0, 100));
							waitKey(0);
							destroyAllWindows();

							Mat testToDisplay;
							_testingData.row(testingIdx).convertTo(testToDisplay, CV_8U);
							namedWindow("correct pred test", WINDOW_AUTOSIZE);
							imshow("correctly classified test", testToDisplay.reshape(0, 100));
							waitKey(0);
							destroyAllWindows();
							numTimesToDisplayCor--;
						}
					}

					else if (numEigenvectors == 100 && numTimesToDisplayIncorr >= 0 && kOfTest == 0) {
						Mat trainToDisplay;
						_data.row(predictedIndex).convertTo(trainToDisplay, CV_8U);
						namedWindow("incorrect pred train", WINDOW_AUTOSIZE);
						imshow("incorrectly classified test", trainToDisplay.reshape(0, 100));
						waitKey(0);
						destroyAllWindows();

						Mat testToDisplay;
						_testingData.row(testingIdx).convertTo(testToDisplay, CV_8U);
						namedWindow("incorrect pred test", WINDOW_AUTOSIZE);
						imshow("incorrectly classified test", testToDisplay.reshape(0, 100));
						waitKey(0);
						destroyAllWindows();
						numTimesToDisplayIncorr--;

					}
				}
			}
		cout << "percentage of correctly predicted images = " << (double)numCorrectlyPredicted*100.0 / testSet.size() << "\n";

		recognitionRateTesting[eigenvectorIdx] += (double)numCorrectlyPredicted*100.0 / testSet.size();
		if (numTimestoDisp > 0 && kOfTest == 0){
			//display mean
			Mat meanToDisplay;
			namedWindow("mean", WINDOW_AUTOSIZE);
			_mean.convertTo(meanToDisplay, CV_8U);
			imshow("mean image", meanToDisplay.reshape(0, 100));
			waitKey(0);
			destroyAllWindows();

			//display first 10 eigenfaces
			for (int i = 0; i < 20; i++){
				double min, max;
				minMaxLoc(_eigenvectors.col(i), &min, &max);
				cout << "min is " << min << '\n';
				cout << "max is" << max << '\n';
				Mat eigenout = Mat(1, 10000, CV_8U);
				_eigenvectors.col(i).convertTo(eigenout, CV_8U, 255.0 / (max - min));
				namedWindow("eigenface", WINDOW_AUTOSIZE);
				imshow("eigenface", eigenout.reshape(0, 100));
				waitKey(0);
				destroyAllWindows();
			}
			numTimestoDisp--;

		}

	
		//display original and reconstructed image
		//display the reconstruction image
		// for 1,5 and 100 eigenvectors display 2 images from training and 2 from test
		if ((numEigenvectors == 1 || numEigenvectors == 5 || numEigenvectors == 3000) &&kOfTest ==0){
			int idx = 7;
			for (int i = 1; i < 3; i++){
				double min, max;
				minMaxLoc(reconstruction, &min, &max);
				cout << "min for reconstruction is " << min << '\n';
				cout << "max for reconstruction is" << max << '\n';
				Mat reconstructionToDisplay;
				reconstruction.row(idx).convertTo(reconstructionToDisplay, CV_8U, 255.0 / (max - min));
				namedWindow("reconstruction training", WINDOW_AUTOSIZE);
				imshow("reconstruction image", reconstructionToDisplay.reshape(0, 100));
				waitKey(0);
				destroyAllWindows();

				//display the original image
				minMaxLoc(_data, &min, &max);
				cout << "min for _data is " << min << '\n';
				cout << "max for _data is" << max << '\n';
				Mat dataToDisplay;
				_data.row(idx).convertTo(dataToDisplay, CV_8U);
				namedWindow("original training", WINDOW_AUTOSIZE);
				imshow("data image", dataToDisplay.reshape(0, 100));
				waitKey(0);
				destroyAllWindows();
				idx += 10;
				minMaxLoc(reconstructionTesting.row(idx), &min, &max);
				cout << "min for reconstruction is " << min << '\n';
				cout << "max for reconstruction is" << max << '\n';
				Mat reconstructionToDisplayTesting;
				reconstructionTesting.row(idx).convertTo(reconstructionToDisplayTesting, CV_8U, 255.0 / (max - min));
				namedWindow("reco testing", WINDOW_AUTOSIZE);
				imshow("reconstruction testig", reconstructionToDisplayTesting.reshape(0, 100));
				waitKey(0);
				destroyAllWindows();

				//display the original image
				minMaxLoc(_testingData.row(idx), &min, &max);
				cout << "min for _data is " << min << '\n';
				cout << "max for _data is" << max << '\n';
				Mat dataToDisplayTesting;
				_testingData.row(idx).convertTo(dataToDisplayTesting, CV_8U);
				namedWindow("ori testing", WINDOW_AUTOSIZE);
				imshow("data testing", dataToDisplayTesting.reshape(0, 100));
				waitKey(0);
				destroyAllWindows();
			}
		}

	}

	//clear training set and test set which will be recreated for next fold
	testSet.clear();
	trainingSet.clear();
	}
	
	//now that we got the reconstruction error just show it
	for (int i = 0; i < 13; i++) {
		cout << "Reconstruction error " << i << " is " << reconstructionErrorsTraining[i]/7<< "\n";
	}

	for (int i = 0; i < 13; i++) {
		cout << "Reconstruction error testing " << i << " is " << reconstructionErrorsTesting[i] / 7 << "\n";
	}

	//now that we got the recognition rate just show it
	for (int i = 0; i < 13; i++) {
		cout << "recognition rate " << i << " is " << recognitionRateTesting[i] / 7 << "\n";
	}

	_mean.convertTo(_mean, CV_8U);
	imshow("mean image", _mean.reshape(0, 100));
	waitKey(0);
	
	
}
//Attempted to do pose estimation but didnt have time
void EigenFaces::PoseEstimation(ImageDataSet data){
	/*
	vector < vector<double>> confusionMatrix;
	vector<vector<Mat>> imagesPerClass;
	vector<Mat> eigenspacePerClass;
	vector<Mat> projectionPerClass;
	vector<Mat> meanPerClass;
	eigenspacePerClass.resize(21);
	projectionPerClass.resize(21);
	imagesPerClass.resize(21);
	//get the classes
	vector<vector<string>> panClasses;
	vector<vector<string >> tiltClasses;
	panClasses.resize(3);
	panClasses[0] = { "-30", "-15" };
	panClasses[1] = { "+0" };
	panClasses[2] = { "+15", "+30" };
	tiltClasses.resize(7);
	tiltClasses[0] = { "-90", "-75" };
	tiltClasses[1] = { "-60", "-45" };
	tiltClasses[2] = { "-30", "-15" };
	tiltClasses[3] = { "+0" };
	tiltClasses[4] = { "+15", "+30" };
	tiltClasses[5] = { "+45", "+60" };
	tiltClasses[6] = { "+75", "+90" };
	//initialize confusion matrix
	confusionMatrix.resize(21);
	for (int i = 0; i < 21; i++){
		confusionMatrix[i].resize(21);
	}

	//go through all pose classes and get the images
	int classIdx = 0;
	for (int i = 0; i < panClasses.size(); i++){
		for (int j = 0; j < tiltClasses.size(); j++){
			//get all the real angles for each class and add them to the imageset for that angle
			for (int k = 0; k < panClasses[i].size(); k++) {
				for (int l = 0; l < tiltClasses[j].size(); j++){
					imagesPerClass[classIdx] = data.QMUL_getImagesByPose(tiltClasses[j][l], panClasses[i][k]);
				}
			}
			classIdx++;
		}
	}

	//now that we have all images train on them
	for (int i = 0; i < 21; i++){
		//create ith eigenspace
		Eigenfaces(imagesPerClass[i]);
		eigenspacePerClass[i].push_back(_eigenvectors);
		Mat projection = project(100, _data);
		projectionPerClass[i].push_back(projection);
		//clear
		_data.deallocate;
		_eigenvectors.deallocate;
		_mean.deallocate;
	}
	*/
	//now get testing images and project each one
}
/*
Projects the input to the eigenspace
Projection = [Image-Mean] * eigenvectors(subset)
*/
Mat EigenFaces::project(int numEigenvectors, Mat input){
	Mat subsetOfEigenvectors = Mat(_eigenvectors.rows, numEigenvectors, CV_32F);
	//get current subset of eigenvectors 

	//now eigenvectors stored by column so get numEienvectors columns TODO: CHECK VALIDITY 
	_eigenvectors(Rect(0, 0, numEigenvectors, _eigenvectors.rows)).copyTo(subsetOfEigenvectors);
	//subsetOfEigenvectors = _eigenvectors;

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

	Mat projection = Mat(input.rows, numEigenvectors, CV_32F);
	//get the projection matrix
	gemm(input, subsetOfEigenvectors, 1.0, Mat(), 0.0, projection);

	cout << "I get projection\n";

	//add the mean matrix again to the data
	for (int i = 0; i < input.rows; i++) {
		add(input.row(i), _mean, input.row(i));
	}


	return projection;
}
//simply makes a prediction based on minimum eukleidian distance of test set projection vs all training projections to eigenspace
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
//Performs reconstruction as Reconstruction = Projection * eigenvectors_transposed
//
Mat EigenFaces::reconstruct(int numEigenvectors, Mat projection){
	Mat subsetOfEigenvectors = Mat(_eigenvectors.rows, numEigenvectors, CV_32F);
	//now eigenvectors stored by column so get numEienvectors columns TODO: CHECK VALIDITY 
	_eigenvectors(Rect(0, 0, numEigenvectors, _eigenvectors.rows)).copyTo(subsetOfEigenvectors);
	//subsetOfEigenvectors = _eigenvectors;


	//ensure all data types are the same
	if (projection.type() != CV_32F || _mean.type() != CV_32F || subsetOfEigenvectors.type() != CV_32F) {
		projection.convertTo(projection, CV_32F);
		_mean.convertTo(_mean, CV_32F);
		subsetOfEigenvectors.convertTo(subsetOfEigenvectors, CV_32F);
		cout << "I make a convesion in reconstruction\n";
	}
	//transpose subset of eigenvectors
	Mat reconstruction = Mat(projection.rows, subsetOfEigenvectors.rows, CV_32F);
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
		add(reconstruction.row(i), _mean, reconstruction.row(i), noArray(), CV_32F);
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
	Mat ret = Mat(height, width, CV_32F);
	for (int i = 0; i < height; i++) {

		src[i].reshape(0, 1).copyTo(ret.row(i));
	}

	return ret;
}