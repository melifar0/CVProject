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

using namespace cv;
using namespace std;

void BOWFaces::BOWcrossValidation(ImageDataSet data) {

	vector<Mat> trainingSet;
	vector<Mat> testSet;

	// test with single pose
	for (int i = 0; i < data.QMUL_SubjectIDs.size(); i++) {
		//get only one pose for testing
		Mat grey = Mat(100, 100, CV_32FC1);
		cvtColor(data.QMUL_getSubjectImageByPose(data.QMUL_SubjectIDs[i], "090", "090"), grey, CV_BGR2GRAY);
		trainingSet.push_back(grey);
	}

	/* Set the number of codewords*/
	const int numCodewords = 20;

	/* Variable definition */
	Mat codeBook;
	vector<vector<Mat>> imageDescriptors;

	/* Training */
	BOWTrain(trainingSet, codeBook, imageDescriptors, numCodewords);

	/* Testing */
	BOWTest(testSet, codeBook, imageDescriptors);
}

void BOWFaces::BOWTrain(const vector<Mat>& imgSet, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords)
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
		vector<Mat> BOWhistrow;
		vector<KeyPoint> keypoints;
		featureDetector->detect(imgSet[i], keypoints);

		Mat BOWhistogram;
		BOWdescriptorExtractor->compute2(imgSet[i], keypoints, BOWhistogram);
		BOWhistrow.push_back(BOWhistogram);
		imageDescriptors.push_back(BOWhistrow);
	}
}

/* Test BoW */
void BOWFaces::BOWTest(const vector<Mat>& imgSet, const Mat &codeBook, const vector<vector<Mat>> &imageDescriptors)
{
	Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> featureExtractor = DescriptorExtractor::create("SIFT");
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
	Ptr<BOWImgDescriptorExtractor> BOWdescriptorExtractor = new BOWImgDescriptorExtractor(featureExtractor, descriptorMatcher);
	BOWdescriptorExtractor->setVocabulary(codeBook);
	vector<int> category_results;
	vector<int> actual_categories;
	//iterate over all training images
	for (int ind = 0; ind < imgSet.size(); ind++) {
		vector<KeyPoint> keypoints;
		featureDetector->detect(imgSet[ind], keypoints);

		Mat BOWhistogram;
		BOWdescriptorExtractor->compute2(imgSet[ind], keypoints, BOWhistogram);
		int bestMatch_category = -1;
		int bestMatch_image = -1;
		double bestMatch_score = -1;
		//iterate over all image descriptors to find best match
		for (int i = 0; i < imageDescriptors.size(); i++) {
			for (int j = 0; j < imageDescriptors[i].size(); j++) {
				if (bestMatch_category == -1 || bestMatch_image == -1 || bestMatch_score == -1) {
					bestMatch_category = i;
					bestMatch_image = j;
					bestMatch_score = compareHist(imageDescriptors[i][j], BOWhistogram, CV_COMP_CHISQR);
				}
				else {
					if (bestMatch_score > compareHist(imageDescriptors[i][j], BOWhistogram, CV_COMP_CHISQR)){
						bestMatch_category = i;
						bestMatch_image = j;
						bestMatch_score = compareHist(imageDescriptors[i][j], BOWhistogram, CV_COMP_CHISQR);
					}
				}
			}
		}
		actual_categories.push_back(ind);
		category_results.push_back(bestMatch_category);
	}

	//compute recognition rate from gathered results
	size_t max_score = category_results.size();
	size_t actual_score = category_results.size();
	for (int k = 0; k < category_results.size(); k++) {
		if (category_results[k] != actual_categories[k]){
			actual_score--;
		}
	}
	double recognition_rate = actual_score / (double)max_score;
	cout << "Recognition rate: " << recognition_rate << endl;
}

