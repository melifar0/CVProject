/*
************************************************
*	A function for LBPand histogram calculation
************************************************
*/

//github test

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

//INPUT: radius, num neighbors, an image
//OUTPUT: matrix of LBP 
void LBP_func(int radius, int numNeighbors, const Mat image)
{

}