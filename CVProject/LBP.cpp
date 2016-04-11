/*
************************************************
*	A function for LBPand histogram calculation
************************************************
*/

//CURRENT VERSION

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
#include "LBP.h"

using namespace cv;
using namespace std;

//INPUT: radius, num neighbors, an image
//OUTPUT: matrix of LBP 

void LBP_class::LBP_func(int radius, int numNeighbors, ImageDataSet data) //set an output
{

	vector<Mat> trainingSet;
	vector<Mat> testSet;

	// REMOVED FOR LOOP FOR TESTING, REPLACE 'data.QMUL_SubjectIDs[1]' with 'data.QMUL_SubjectIDs[i]'
	//for (int i = 0; i < data.QMUL_SubjectIDs.size(); i++) {

		//get only one pose for testing
		Mat grey = Mat(100, 100, CV_32FC1);
		cvtColor(data.QMUL_getSubjectImageByPose(data.QMUL_SubjectIDs[1], "090", "090"), grey, CV_BGR2GRAY);
		trainingSet.push_back(grey);
	//}

	//Variables to hold the result of the '>' comparison between the center pixel and the surrounding ones
	int LBP_1, LBP_2, LBP_3, LBP_4, LBP_5, LBP_6, LBP_7, LBP_8, LBP_9,
		LBP_10, LBP_11, LBP_12, LBP_13, LBP_14, LBP_15, LBP_16 = 0;
	//variable to count the changes to check for uniformity
	int changes = 0;

	//!!!!!!!!!!!!!!!CHANGED ALL 'image' TO 'trainingSet[j]'!!!!!!!!!!!!!

	int j = 0; //FOR TESTING

	// -2 and so output matrix is proper size. Cannot compute LBP for edge pixels. Would be out of Bounds 
	Mat LBP_mat8 = Mat::zeros(trainingSet[j].rows - 2, trainingSet[j].cols - 2, CV_32SC1);
	//-4 because two pixels are taken off of each edge
	Mat LBP_mat16 = Mat::zeros(trainingSet[j].rows - 4, trainingSet[j].cols - 4, CV_32SC1);

	//cout << "After mat declarations, after color conversion" << '\n';
	//cout << "trainingSet[j].rows - 2 = " << (trainingSet[j].rows - 2) << '\n';
	//cout << "trainingSet[j].rows - 4 = " << (trainingSet[j].rows - 4) << '\n';


	//going to hard code for operators [1,8], [2,8], [2,16]
	if (radius == 1)
	{
		if (numNeighbors == 8)
		{
			//The radius is 1, for loop will exclude outer most pixels to avoid out of bounds.
			for (int rowIDX = 1; rowIDX < (trainingSet[j].rows - 1); rowIDX++)
			{
				for (int colIDX = 1; colIDX < (trainingSet[j].cols - 1); colIDX++)
				{
					cout << "Inside the for loop" << '\n';
					if (trainingSet[j].type() != CV_32F) {
						cout << "training set does not have the right type \n";
						trainingSet[j].convertTo(trainingSet[j], CV_32F);

					}
					cout << "dimesions of training set = " << trainingSet[j].rows << "x" << trainingSet[j].cols << "\n";
					float centerPX = trainingSet[j].at<float>(rowIDX, colIDX);
					cout << "center pixel value is: " << centerPX << '\n';
					
					//starting from the top left corner moving clockwise. c is center pixel, 0,0
					//_____________
					//|	  |   |   |
					//| 1 | 2 | 3 |
					//|___|___|___|
					//|   |   |   |
					//| 8 | c | 4 |
					//|___|___|___|
					//|   |   |   |
					//| 7 | 6 | 5 |
					//|___|___|___|
					//
					//moving clockwise, 1 if the square is > center pixel value.
					LBP_1 = (trainingSet[j].at<float>(rowIDX - 1, colIDX - 1) > centerPX) ? 1 : 0;
					LBP_2 = (trainingSet[j].at<float>(rowIDX - 1, colIDX)     > centerPX) ? 1 : 0;
					LBP_3 = (trainingSet[j].at<float>(rowIDX - 1, colIDX + 1) > centerPX) ? 1 : 0;
					LBP_4 = (trainingSet[j].at<float>(rowIDX, colIDX + 1)     > centerPX) ? 1 : 0;
					LBP_5 = (trainingSet[j].at<float>(rowIDX + 1, colIDX + 1) > centerPX) ? 1 : 0;
					LBP_6 = (trainingSet[j].at<float>(rowIDX + 1, colIDX)     > centerPX) ? 1 : 0;
					LBP_7 = (trainingSet[j].at<float>(rowIDX + 1, colIDX - 1) > centerPX) ? 1 : 0;
					LBP_8 = (trainingSet[j].at<float>(rowIDX, colIDX - 1)     > centerPX) ? 1 : 0;

					cout << "LBP_1 is: " << LBP_1 << '\n';
					cout << "LBP_2 is: " << LBP_2 << '\n';
					cout << "LBP_3 is: " << LBP_3 << '\n';
					cout << "LBP_4 is: " << LBP_4 << '\n';
					cout << "LBP_5 is: " << LBP_5 << '\n';
					cout << "LBP_6 is: " << LBP_6 << '\n';
					cout << "LBP_7 is: " << LBP_7 << '\n';
					cout << "LBP_8 is: " << LBP_8 << '\n';

					//turn the binary digits into a decimal number
					int LBP_r1n8 = (LBP_1 * 2^7) + (LBP_2 * 2^6) + (LBP_3 * 2^5) + (LBP_4 * 2^4) + 
								   (LBP_5 * 2^3) + (LBP_6 * 2^2) + (LBP_7 * 2^1) + (LBP_8 * 2^0);

					cout << "LBP_r1n8 is: " << LBP_r1n8 << '\n';

					//now check if it is a uniform LBP or not by counting the changes in bits. 
					//if more than two, then not uniform
					int binNum[8] = { LBP_1, LBP_2, LBP_3, LBP_4, LBP_5, LBP_6, LBP_7, LBP_8 };
					changes = 0;
					for (int i = 0; i < (8-1); i++)
					{
						if (binNum[i] != binNum[i + 1])
						{
							changes++;
						}
					}
					cout << "changes is: " << changes << '\n';
					//all non uniform numbers get assigned arbitrary number 10101010 = 170
					if (changes > 2)
					{
						LBP_r1n8 = 170;
					}

					cout << "LBP_r1n8 is: " << LBP_r1n8 << '\n';

					cout << "rowIDX -1 = " << rowIDX-1 << '\n';
					cout << "colIDX -1 = " << colIDX-1 << '\n';

					//set the output matrix at this pixel index as the value of the pixel LBP
					LBP_mat8.at<int>(rowIDX - 1, colIDX - 1) = LBP_r1n8; // -1 to ensure we start filling it up from (0,0)

					/*
					//A DIFFERENT METHOD WITH MORE BIT LOGIC
					uchar centerPX = trainingSet[j].at<uchar>(rowIDX, colIDX);
					unsigned char LBP_bNum = 0;
					//create the binary number from the bool PX > center. This is the LBP
					LBP_bNum |= (trainingSet[j].at<uchar>(rowIDX - 1, colIDX - 1) > centerPX) << 7;
					LBP_bNum |= (trainingSet[j].at<uchar>(rowIDX - 1, colIDX    ) > centerPX) << 6;
					LBP_bNum |= (trainingSet[j].at<uchar>(rowIDX - 1, colIDX + 1) > centerPX) << 5;
					LBP_bNum |= (trainingSet[j].at<uchar>(rowIDX    , colIDX + 1) > centerPX) << 4;
					LBP_bNum |= (trainingSet[j].at<uchar>(rowIDX + 1, colIDX + 1) > centerPX) << 3;
					LBP_bNum |= (trainingSet[j].at<uchar>(rowIDX + 1, colIDX    ) > centerPX) << 2;
					LBP_bNum |= (trainingSet[j].at<uchar>(rowIDX + 1, colIDX - 1) > centerPX) << 1;
					LBP_bNum |= (trainingSet[j].at<uchar>(rowIDX    , colIDX - 1) > centerPX) << 0;
					*/
				}
			}
		}
		else
		{
			cout << "Function input 'numNeighbors' is: %d. This function only accepts operators [1,8], [2,16], [2,8]" << numNeighbors << '\n';
		}
	}
	/*
	else if (radius == 2)
	{
		if (numNeighbors == 16)
		{
			//The radius is 2, for loop will exclude outer most 2 pixels to avoid out of bounds.
			for (int rowIDX = 2; rowIDX < (trainingSet[j].rows - 2); rowIDX++)
			{
				for (int colIDX = 2; colIDX < (trainingSet[j].cols - 2); colIDX++)
				{

					float centerPX = trainingSet[j].at<float>(rowIDX, colIDX);

					//starting from the top left corner moving clockwise. c is center pixel, 0,0
					//_________________________
					//|	   |    |    |    |    |
					//|    |  3 |  4 |  5 |    |
					//|____|____|____|____|____|
					//|    |    |    |    |    |
					//|  1 |  2 |    |  6 |  7 |
					//|____|____|____|____|____|
					//|    |    |    |    |    |
					//| 16 |    | C  |    |  8 |
					//|____|____|____|____|____|
					//|    |    |    |    |    |
					//| 15 | 14 |    | 10 | 9  |
					//|____|____|____|____|____|
					//|    |    |    |    |    |
					//|    | 13 | 12 | 11 |    |
					//|____|____|____|____|____|
					//

					//moving clockwise, 1 if the square is > center pixel value.
					LBP_1  = (trainingSet[j].at<float>(rowIDX - 1, colIDX - 2) > centerPX) ? 1 : 0;
					LBP_2  = (trainingSet[j].at<float>(rowIDX - 1, colIDX - 1) > centerPX) ? 1 : 0;
					LBP_3  = (trainingSet[j].at<float>(rowIDX - 2, colIDX - 1) > centerPX) ? 1 : 0;
					LBP_4  = (trainingSet[j].at<float>(rowIDX - 2, colIDX)     > centerPX) ? 1 : 0;
					LBP_5  = (trainingSet[j].at<float>(rowIDX - 2, colIDX + 1) > centerPX) ? 1 : 0;
					LBP_6  = (trainingSet[j].at<float>(rowIDX - 1, colIDX + 1) > centerPX) ? 1 : 0;
					LBP_7  = (trainingSet[j].at<float>(rowIDX - 1, colIDX + 2) > centerPX) ? 1 : 0;
					LBP_8  = (trainingSet[j].at<float>(rowIDX,	   colIDX + 2) > centerPX) ? 1 : 0;
					LBP_9  = (trainingSet[j].at<float>(rowIDX + 1, colIDX + 2) > centerPX) ? 1 : 0;
					LBP_10 = (trainingSet[j].at<float>(rowIDX + 1, colIDX + 1) > centerPX) ? 1 : 0;
					LBP_11 = (trainingSet[j].at<float>(rowIDX + 2, colIDX + 1) > centerPX) ? 1 : 0;
					LBP_12 = (trainingSet[j].at<float>(rowIDX + 2, colIDX)     > centerPX) ? 1 : 0;
					LBP_13 = (trainingSet[j].at<float>(rowIDX + 2, colIDX - 1) > centerPX) ? 1 : 0;
					LBP_14 = (trainingSet[j].at<float>(rowIDX + 1, colIDX - 1) > centerPX) ? 1 : 0;
					LBP_15 = (trainingSet[j].at<float>(rowIDX + 1, colIDX - 2) > centerPX) ? 1 : 0;
					LBP_16 = (trainingSet[j].at<float>(rowIDX,     colIDX - 2) > centerPX) ? 1 : 0;

					//turn the binary digits into a decimal number
					int LBP_r2n16 = (LBP_1 * 2^15) + (LBP_2 * 2^14) + (LBP_3 * 2^13) + (LBP_4 * 2^12) + 
							  (LBP_5 * 2^11) + (LBP_6 * 2^10) + (LBP_7 * 2^9)  + (LBP_8 * 2^8)  +
							  (LBP_9 * 2^7)  + (LBP_10 * 2^6) + (LBP_11 * 2^5) + (LBP_12 * 2^4) + 
							  (LBP_13 * 2^3) + (LBP_14 * 2^2) + (LBP_15 * 2^1) + (LBP_16 * 2^0);

					//now check if it is a uniform LBP or not by counting the changes in bits. 
					//if more than two, then not uniform
					int binNum16[16] = { LBP_1, LBP_2, LBP_3, LBP_4, LBP_5, LBP_6, LBP_7, LBP_8, LBP_9,
									   LBP_10, LBP_11, LBP_12, LBP_13, LBP_14, LBP_15, LBP_16 };
					changes = 0;
					for (int i = 0; i < (16-1); i++)
					{
						if (binNum16[i] != binNum16[i + 1])
						{
							changes++;
						}
					}
					//all non uniform numbers get assigned arbitrary number 1010101010101010 = 43690
					if (changes > 2)
					{
						LBP_r2n16 = 43690;
					}

					//set the output matrix at this pixel index as the value of the pixel LBP
					LBP_mat16.at<int>(rowIDX - 2, colIDX - 2) = LBP_r2n16; // -2 to ensure we start filling it up from (0,0)
				}
			}
		}
		else if (numNeighbors == 8)
		{
			for (int rowIDX = 2; rowIDX < (trainingSet[j].rows - 2); rowIDX++)
			{
				for (int colIDX = 2; colIDX < (trainingSet[j].cols - 2); colIDX++)
				{
					//WHAT IS THE PIXEL TYPE FROM TEH trainingSet[j]S??
					float centerPX = trainingSet[j].at<float>(rowIDX, colIDX);

					//starting from the top left corner, c is center pixel, 0,0
					//_________________________
					//|	   |    |    |    |    |
					//|    |    |  2 |    |    |
					//|____|____|____|____|____|
					//|    |    |    |    |    |
					//|    |  1 |    |  3 |    |
					//|____|____|____|____|____|
					//|    |    |    |    |    |
					//|  8 |    | C  |    |  4 |
					//|____|____|____|____|____|
					//|    |    |    |    |    |
					//|    |  7 |    |  5 |    |
					//|____|____|____|____|____|
					//|    |    |    |    |    |
					//|    |    |  6 |    |    |
					//|____|____|____|____|____|
					//
					//moving clockwise, 1 if the square is > center pixel value.
					int LBP_1 = (trainingSet[j].at<float>(rowIDX - 1, colIDX - 1) > centerPX) ? 1 : 0;
					int LBP_2 = (trainingSet[j].at<float>(rowIDX - 2, colIDX)     > centerPX) ? 1 : 0;
					int LBP_3 = (trainingSet[j].at<float>(rowIDX - 1, colIDX + 1) > centerPX) ? 1 : 0;
					int LBP_4 = (trainingSet[j].at<float>(rowIDX,	  colIDX + 2) > centerPX) ? 1 : 0;
					int LBP_5 = (trainingSet[j].at<float>(rowIDX + 1, colIDX + 1) > centerPX) ? 1 : 0;
					int LBP_6 = (trainingSet[j].at<float>(rowIDX + 2, colIDX)     > centerPX) ? 1 : 0;
					int LBP_7 = (trainingSet[j].at<float>(rowIDX + 1, colIDX - 1) > centerPX) ? 1 : 0;
					int LBP_8 = (trainingSet[j].at<float>(rowIDX,	  colIDX - 2) > centerPX) ? 1 : 0;

					//turn the binary digits into a decimal number
					int LBP_r2n8 = (LBP_1 * 2 ^ 7) + (LBP_2 * 2 ^ 6) + (LBP_3 * 2 ^ 5) + (LBP_4 * 2 ^ 4) +
							  (LBP_5 * 2 ^ 3) + (LBP_6 * 2 ^ 2) + (LBP_7 * 2 ^ 1) + (LBP_8 * 2 ^ 0);

					//now check if it is a uniform LBP or not by counting the changes in bits.
					//if more than two, then not uniform
					int binNum8[8] = { LBP_1, LBP_2, LBP_3, LBP_4, LBP_5, LBP_6, LBP_7, LBP_8 };
					int changes = 0;
					for (int i = 0; i < (8-1); i++)
					{
					if (binNum8[i] != binNum8[i + 1])
					{
					changes++;
					}
					}
					//all non uniform numbers get assigned arbitrary number 10101010 = 170
					if (changes > 2)
					{
						LBP_r2n8 = 170;
					}

					//set the output matrix at this pixel index as the value of the pixel LBP
					LBP_mat16.at<int>(rowIDX - 2, colIDX - 2) = LBP_r2n8; // -2 to ensure we start filling it up from (0,0)
				}
			}
		}
		else
		{
			cout << "Function input 'numNeighbors' is: %d. This function only accepts operators [1,8], [2,16], [2,8]" << numNeighbors << '\n';
		}
	}
	*/
	else
	{
		cout << "Function input 'radius' is: %d. This function only accepts operators [1,8], [2,16], [2,8]" << radius << '\n';
	}

	cout << "Reached the end before the print statements." << '\n';

	Mat LBP_mattest1 = Mat::zeros(trainingSet[j].rows - 2, trainingSet[j].cols - 2, CV_8U);
	Mat LBP_mattest2 = Mat::zeros(trainingSet[j].rows - 4, trainingSet[j].cols - 4, CV_8U);

	LBP_mat8.convertTo(LBP_mattest1, CV_8U);
	imshow("test image [1,8]", LBP_mattest1);
	waitKey(0);

	LBP_mat16.convertTo(LBP_mattest2, CV_8U);
	imshow("test image [2,16] or [2,8]", LBP_mattest2);
	waitKey(0);
}