
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "ImageDataSet.h"


using namespace cv;
using namespace std;




//7 fold cross validation 

class KFold {

	const int k = 7;
	vector<vector<Mat>> partitionedData;
	vector<int> indicesToConsider;


	/*this is the only function you should call it returns a vector containing 7 Mat vectors of partitioned data*/
public:vector<vector<Mat>> getPartitionedData(ImageDataSet data){
		   initializeIndices();
		   initializePartitionedData();
		   createFolds(data);
		   cout << "I return partitioned data \n";
		   return partitionedData;
}

	   void initializePartitionedData(){
		   partitionedData.resize(k);
	   }
	   void initializeIndices() {
		   for (int i = 0; i < k; i++) {
			   indicesToConsider.push_back(i);
		   }
	   }
	   //Partitions the data in 7 disjoint subsets
	   void createFolds(ImageDataSet data){

		   int randomIndex = 0;
		   for (int i = 0; i < data.QMUL_SubjectIDs.size(); i++) {

			   for (int j = 0; j < data.QMUL_PanCodes.size(); j++) {

				   for (int k = 0; k < data.QMUL_TiltCodes.size(); k++){
					   //get a random index in which to place the image
					   randomIndex = rand() % indicesToConsider.size();
					   randomIndex = indicesToConsider.at(randomIndex);
					   //get sequence of PanCodes and Tilt Codes to skip
					   Mat grey = Mat(100, 100, CV_32FC1);
					   cvtColor(data.QMUL_getSubjectImageByPose(data.QMUL_SubjectIDs[i], data.QMUL_TiltCodes[k], data.QMUL_PanCodes[j]), grey, CV_BGR2GRAY);
					   partitionedData.at(randomIndex).push_back(grey);

					   //check if we have saturation at given random index and if so remove it from
					   //indices to consider, given pseudorandom number generation which is not uniform
					   if (partitionedData.at(randomIndex).size() >= 589) {
						   indicesToConsider.erase(remove(indicesToConsider.begin(), indicesToConsider.end(), randomIndex), indicesToConsider.end());
					   }
				   }
			   }

		   }
	   }
};


void crossValidation(ImageDataSet data);

void main(void)
{
	// Initialize OpenCV nonfree module
	initModule_nonfree();

	//Put the full path of the QMUL Multiview Face Dataset folder here
	//C:\Users\Dimitri-G\CVProject\CVProject\resources\HeadPoseImageDatabase
	const string QMULPath = "C:/Users/Dimitri-G/CVProject/CVProject/resources/QMUL_360degreeViewSphere_FaceDatabase/Set1_Greyscale";
	int k = 0;
	//Put the full path of the Head Pose Image Database folder here
	const string HeadPoseDBPath = "C:/Users/Dimitri-G/CVProject/CVProject/resources/HeadPoseImageDatabase";

	//Load the dataset by instantiating the helper class
	ImageDataSet data(QMULPath, HeadPoseDBPath);

	//Terminate if data is not successfully loaded
	if (!data.isSuccessfullyLoaded())
	{
		cout << "An error occurred loading the dataset, press Enter to exit" << endl;
		getchar();
		return;
	}

	//Question 1.1
	data.QMUL_displaySubjectImages("YongminY");
	//Question 1.2
	data.HPDB_displayImagesBySubjectIDandSerie("Person15", "2");

	/*ALL OF THE CODE BELOW IS EXAMPLES, DELETE ONCE DONE*/
	/* QMUL data and functions start with QMUL_*  */
	//All important QMUL data
	data.QMUL_SubjectIDs;
	data.QMUL_Images;
	data.QMUL_TiltCodes;
	data.QMUL_PanCodes;
	/* HPDB data and functions start with HPDB_*  */
	//All important HPDB data
	data.HPDB_SubjectIDs;
	data.HPDB_Images;
	data.HPDB_Labels;
	data.HPDB_Series;
	data.HPDB_TiltCodes;
	data.HPDB_PanCodes;

	//Examples of how to display individual images
	imshow("Test1", data.QMUL_getSubjectImageByPose("AdamB", "090", "090"));
	waitKey(0);
	destroyWindow("Test1");
	imshow("Test2", data.HPDB_getSubjectImageByPoseAndSerie("Person01", "1", "-15", "+0"));
	waitKey(0);
	destroyWindow("Test2");

	//implement the rest of the code here

	crossValidation(data);
}

/*
Input: Set of training Images
Output: eigenvectors and eigenvalues, ie principle components of data
Functionality:
-Creates data matrix, each row containing vectorized training image
-Computes the covariance matrix
-Sets the values of eigenvectors and eigenvalues
*/
Mat _eigenvectors;
Mat _eigenvalues;
Mat _mean;
Mat _mean2;
void Eigenfaces(const vector<Mat>& imgSet) {
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
	Mat data = Mat(height, width, imgSet[0].type());
	//create data matrix
	for (int i = 0; i < height; i++) {
		if (imgSet[i].isContinuous()){
			imgSet[i].reshape(0, 1).convertTo(data.row(i), data.type(), 1, 0);
		}
		else {
			imgSet[i].clone().reshape(0, 1).copyTo(data.row(i));
		}
	}
	vector<Scalar> mean;
	//go through the columns and get the mean

	for (int i = 0; i < width; i++) {
		Scalar temp = cv::mean(data.col(i));
		mean.push_back(temp);
	}
	_mean2 = Mat(mean);
	//subtract mean from each row
	/*
	for (int i = 0; i < data.rows; i++) {
	data.row(i) = data.row(i) - mean[i];
	}*/

	//get the covariance matrix
	Mat newData;
	Mat S(10000, 10000, CV_32FC1);
	if (data.type() != CV_32FC1 || data.type() != CV_64FC1 || data.type() != CV_32FC2 || data.type() != CV_64FC2) {
		cout << "format is =  \n" << data.type() << '\n';
		data.convertTo(newData, CV_32FC1);
		printf("Matrix does not have right format\n");
		cout << "format is = " << newData.type() << '\n';
		cout << "format is = " << CV_32FC1 << '\n';
}
	Mat cov;
	_mean = Mat(1, data.cols, CV_32F, 0);
	calcCovarMatrix(data, cov, _mean, CV_COVAR_NORMAL + CV_COVAR_ROWS);
	Mat eigenvalues;
	Mat eigenvectors;

	bool ret;
	cout << "I get covariance matrix \n";
	ret = eigen(cov, _eigenvalues, _eigenvectors);

}

void crossValidation(ImageDataSet data) {

	//Do 7 fold cross validation
	//Create 7 sets of equal size, 6 sets are the training sets and 1 set is the testing set
	/*
	vector<Mat> trainingSet;
	for (int i = 0; i < data.QMUL_SubjectIDs.size(); i++) {
	for (int j = 0; j < data.QMUL_PanCodes.size(); j++) {
	for (int k = 0; k < data.QMUL_TiltCodes.size(); k++){
	//get sequence of PanCodes and Tilt Codes to skip
	Mat grey = Mat(100, 100, CV_32FC1);
	cvtColor(data.QMUL_getSubjectImageByPose(data.QMUL_SubjectIDs[i], "090", "090"), grey, CV_BGR2GRAY);
	trainingSet.push_back(grey);
	}
	}

	}*/

	vector<Mat> trainingSet;
	vector<Mat> testSet;
	KFold kfold;
	vector<vector<Mat>> partitioned = kfold.getPartitionedData(data);
	testSet.insert(testSet.end(), partitioned[0].begin(), partitioned[0].end());
	for (int i = 1; i < 7; i++){
		trainingSet.insert(trainingSet.end(), partitioned[i].begin(), partitioned[i].end());
	}

	cout << "size of training set =" << trainingSet.size() << '\n';
	cout << "size of test set =" << testSet.size() << '\n';


	//call the function to produce eigenvectors
	Eigenfaces(trainingSet);
	//display eigenvalues
	vector<int> eigenvectorsTest = { 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, _eigenvalues.rows };
	/*
	for (int i = 0; i < eigenvectorsTest.size(); i++){
	int numEigenvalues = eigenvectorsTest[i];

	Mat subsetOfEigenvectors;
	//get current subset of eigenvectors
	_eigenvectors(Rect(0, 0, _eigenvalues.cols, numEigenvalues)).copyTo(subsetOfEigenvectors);
	//do the projection
	Mat projection = (_data - _mean)*

	}*/
	cout << _mean;
	cout << _mean.rows;
	cout << _mean.cols;
	//cout << _mean2;

	//display the average image
	imshow("average image", _mean.reshape(0, data.QMUL_getSubjectImageByPose(data.QMUL_SubjectIDs[0], "090", "090").rows));
	waitKey(0);
	//display avg image after converting

	//display the average image
	imshow("average image", _mean2.reshape(0, data.QMUL_getSubjectImageByPose(data.QMUL_SubjectIDs[0], "090", "090").rows));
	waitKey(0);
}


