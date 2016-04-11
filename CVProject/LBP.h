#ifndef LBP_H
#define LBP_H

#include <opencv2/opencv.hpp>
#include <string>
#include "ImageDataSet.h"

using namespace cv;
using namespace std;

class LBP_class
{
public:
void LBP_func(int radius, int numNeighbors, ImageDataSet data);
};

//void LBP_func(int radius, int numNeighbors, ImageDataSet data);

#endif
