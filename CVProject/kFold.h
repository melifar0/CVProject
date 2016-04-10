#ifndef KFOLD_H
#define KFOLD_H
#endif
#include <opencv2/opencv.hpp>
#include <string>
#include "ImageDataSet.h"
class KFold {

public:
	vector<vector<Mat>> getPartitionedData(ImageDataSet data);
	void createFolds(ImageDataSet data);
	void initializePartitionedData();
	void initializeIndices();
	void initializeLabels();
	vector<vector<string>> getLabels();


};

