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
using namespace cv;
using namespace std;

	const int k = 7;
	vector<vector<Mat>> partitionedData;
	vector<int> indicesToConsider;

	/*this is the only function you should call it returns a vector containing 7 Mat vectors of partitioned data*/
      vector<vector<Mat>> KFold::getPartitionedData(ImageDataSet data){
		   initializeIndices();
		   initializePartitionedData();
		   createFolds(data);
		   cout << "I return partitioned data \n";
		   return partitionedData;
}

	  void KFold::initializePartitionedData(){
		   partitionedData.resize(k);
	   }
	   void KFold::initializeIndices() {
		   for (int i = 0; i < k; i++) {
			   indicesToConsider.push_back(i);
		   }
	   }
	   //Partitions the data in 7 disjoint subsets
	   void KFold::createFolds(ImageDataSet data){

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


