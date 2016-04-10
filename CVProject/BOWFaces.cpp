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
#include "BOWFaces.h"
#include <numeric>

using namespace cv;
using namespace std;
ofstream myfile;

void BOWFaces::BOWcrossValidation(ImageDataSet data) {

	vector<Mat> trainingSet;
	vector<Mat> testSet;
	vector<string> trainingSetLabels;
	vector<string> testSetLabels;
	double recogRates[7];
	double avgRecogRate;

	myfile.open("BOWFacesOut.txt", fstream::app);

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

		/*// test with single pose
		for (int i = 0; i < data.QMUL_SubjectIDs.size(); i++) {
		//get only one pose for testing
		Mat grey = Mat(100, 100, CV_32FC1);
		cvtColor(data.QMUL_getSubjectImageByPose(data.QMUL_SubjectIDs[i], "090", "090"), grey, CV_BGR2GRAY);
		trainingSet.push_back(grey);
		}*/

		/* Set the number of codewords*/
		const int numCodewords = 50; // need to check for all 13 different values

		/* Variable definition */
		Mat codeBook;
		vector<Mat> BOWhistrow;

		/* Training */
		BOWTrain(trainingSet, codeBook, BOWhistrow , numCodewords);

		/* Testing */
		recogRates[kOfTest] = BOWTest(testSet, codeBook, BOWhistrow, trainingSetLabels, testSetLabels);
		testSet.clear();
		trainingSet.clear();
	}

	double sum = 0;
	for (int i = 0; i < 7; i++){
		sum += recogRates[i];
	}
	avgRecogRate = ((double)sum) / 7.0;
	cout << "Average Recognition rate: " << avgRecogRate << endl;
	myfile << "Average Recognition rate: " << avgRecogRate << "\n";
	system("pause");
	myfile.close();
}

void BOWFaces::BOWTrain(const vector<Mat>& imgSet, Mat &codeBook, vector<Mat> &BOWhistrow, const int numCodewords)
{
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> featureExtractor = DescriptorExtractor::create("SIFT");
	Mat allDescriptors;
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
	Ptr<BOWImgDescriptorExtractor> BOWdescriptorExtractor = new BOWImgDescriptorExtractor(featureExtractor, descriptorMatcher);
	//iterate over all training images
	for (int i = 0; i < imgSet.size(); i++) {
		vector<KeyPoint> keypoints;
		featureDetector->detect(imgSet[i], keypoints);

		Mat descriptors;
		featureExtractor->compute(imgSet[i], keypoints, descriptors);
		allDescriptors.push_back(descriptors);
	}
	BOWKMeansTrainer bowTrainer(numCodewords);
	bowTrainer.add(allDescriptors);
	codeBook = bowTrainer.cluster();

	BOWdescriptorExtractor->setVocabulary(codeBook);
	//iterate over all training images

	for (int i = 0; i < imgSet.size(); i++) {
		vector<KeyPoint> keypoints;
		featureDetector->detect(imgSet[i], keypoints);

		Mat BOWhistogram;
		BOWdescriptorExtractor->compute2(imgSet[i], keypoints, BOWhistogram);
		BOWhistrow.push_back(BOWhistogram);
	}
}

/* Test BoW */
double BOWFaces::BOWTest(const vector<Mat>& imgSet, const Mat &codeBook, const vector<Mat> &BOWhistrow, const vector<string> &trainingSetLabels, const vector<string> &testSetLabels)
{
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> featureExtractor = DescriptorExtractor::create("SIFT");
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
	Ptr<BOWImgDescriptorExtractor> BOWdescriptorExtractor = new BOWImgDescriptorExtractor(featureExtractor, descriptorMatcher);
	BOWdescriptorExtractor->setVocabulary(codeBook);
	int matches = 0;

	//iterate over all test images
	for (int ind = 0; ind < imgSet.size(); ind++) {
		vector<KeyPoint> keypoints;
		featureDetector->detect(imgSet[ind], keypoints);

		Mat BOWhistogram;
		BOWdescriptorExtractor->compute2(imgSet[ind], keypoints, BOWhistogram);
		int bestMatch_image = -1;
		double bestMatch_score = -1;
		//iterate over all image descriptors to find best match
		for (int i = 0; i < BOWhistrow.size(); i++) {
			double curScore = chiSquareDist(BOWhistrow[i], BOWhistogram);
			if (bestMatch_image == -1 || bestMatch_score == -1) {
				bestMatch_image = i;
				bestMatch_score = curScore;
			}
			else {
				if (bestMatch_score > curScore){
					bestMatch_image = i;
					bestMatch_score = curScore;
				}
			}
		}
		myfile << ind << ", " << bestMatch_image << "\n";
		string actual_image = testSetLabels[ind];
		string image_result = trainingSetLabels[bestMatch_image];
		if (actual_image.compare(image_result)){
			matches++;
		}
	}

	//compute recognition rate from gathered results
	size_t max_score = imgSet.size();

	double recognition_rate = ((double)matches) / (double)max_score;
	cout << "Actual score: " << matches << endl;
	cout << "Max score: " << max_score << endl;
	cout << "Recognition rate: " << recognition_rate << endl;
	myfile << "Recognition rate: " << recognition_rate << "\n";
	return recognition_rate;
}


// This method is adapted from the OpenCV function compareHist() found in ..\opencv\sources\modules\imgproc\src\histogram.cpp
double BOWFaces::chiSquareDist(InputArray _H1, InputArray _H2)
{
	Mat H1 = _H1.getMat();
	Mat	H2 = _H2.getMat();
	/*float chiSquareDistance = 0;

	for (int i = 0; i < H1.cols; i++) {
		float diff = (H1.at<float>(0, i)) - (H2.at<float>(0, i));
		float sum = (H1.at<float>(0, i)) + (H2.at<float>(0, i));
		if (fabs(sum) > DBL_EPSILON)
			chiSquareDistance += (float)(pow(diff, 2) / sum);
	}
	return chiSquareDistance;*/
	const Mat* arrays[] = { &H1, &H2, 0 };
	Mat dims[2];
	NAryMatIterator iterator(arrays, dims);
	double result = 0;

	for (size_t i = 0; i < iterator.nplanes; i++, ++iterator)
	{
		const float* h1 = (const float*)iterator.planes[0].data;
		const float* h2 = (const float*)iterator.planes[1].data;
		int len = iterator.planes[0].rows*iterator.planes[0].cols;

		for (int j = 0; j < len; j++)
		{
			double a = h1[j] - h2[j];
			double b = h1[j] + h2[j];
			if (fabs(b) > DBL_EPSILON)
				result += a*a / b;
		}
	}
	return result;
}
