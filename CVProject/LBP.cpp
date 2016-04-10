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
#include "LBP.h"

using namespace cv;
using namespace std;

//INPUT: radius, num neighbors, an image
//OUTPUT: matrix of LBP 
void LBP::LBP_func(int radius, int numNeighbors, const Mat image) //set an output
{
	//going to hard code for operators [1,8], [2,8], [2,16]
	if (radius == 1)
	{
		//The radius is one, Mat of LBP will exclude outer most pixels to avoid out of bounds.
		Mat LBP_mat = Mat::zeros(image.rows - 2, image.cols - 2, CV_8UC1);
		if (numNeighbors == 8)
		{
			for (int rowIDX = 1; rowIDX < (image.rows - 1); rowIDX++)
			{
				for (int colIDX = 1; colIDX < (image.cols - 1); colIDX++)
				{
					//WHAT IS THE PIXEL TYPE FROM TEH IMAGES??
					int centerPX = image.at</*CHECK THIS*/int>(rowIDX, colIDX); 
					
					//starting from the top left corner, c is center pixel, 0,0
					//_____________
					//|	  |   |   |
					//| 0 | 1 | 2 |
					//|___|___|___|
					//|   |   |   |
					//| 7 | c | 3 |
					//|___|___|___|
					//|   |   |   |
					//| 6 | 5 | 4 |
					//|___|___|___|
					//
					//moving clockwise, 1 if the square is > center pixel value.
					int LBP_1 = (image.at<uchar>(rowIDX - 1, colIDX - 1) > centerPX) ? 1 : 0;
					int LBP_2 = (image.at<uchar>(rowIDX - 1, colIDX)     > centerPX) ? 1 : 0;
					int LBP_3 = (image.at<uchar>(rowIDX - 1, colIDX + 1) > centerPX) ? 1 : 0;
					int LBP_4 = (image.at<uchar>(rowIDX, colIDX + 1)     > centerPX) ? 1 : 0;
					int LBP_5 = (image.at<uchar>(rowIDX + 1, colIDX + 1) > centerPX) ? 1 : 0;
					int LBP_6 = (image.at<uchar>(rowIDX + 1, colIDX)     > centerPX) ? 1 : 0;
					int LBP_7 = (image.at<uchar>(rowIDX + 1, colIDX - 1) > centerPX) ? 1 : 0;
					int LBP_8 = (image.at<uchar>(rowIDX, colIDX - 1)     > centerPX) ? 1 : 0;

					//turn the binary digits into a decimal number
					int LBP = (LBP_1 * 2^7) + (LBP_2 * 2^6) + (LBP_3 * 2^5) + (LBP_4 * 2^4) + (LBP_5 * 2^3) + (LBP_6 * 2^2) + (LBP_7 * 2^1) + (LBP_8 * 2^0);

					//now check if it is a uniform LBP or not by counting the changes in bits. 
					//if more than two, then not uniform
					int binNum[8] = { LBP_1, LBP_2, LBP_3, LBP_4, LBP_5, LBP_6, LBP_7, LBP_8 };
					int changes = 0;
					for (int i = 0; i < (8 - 1); i++)
					{
						if (binNum[i] != binNum[i + 1])
						{
							changes++;
						}
					}
					//all non uniform numbers get assigned arbitrary number 10101010 = 170
					if (changes > 2)
					{
						int LBP = 170;
					}

					//set the output matrix at this pixel index as the value of the pixel LBP
					LBP_mat.at<uchar>(rowIDX-1, colIDX-1) = LBP; // -1 to ensure we start filling it up from (0,0)

					/*
					//A DIFFERENT METHOD WITH MORE BIT LOGIC
					uchar centerPX = image.at<uchar>(rowIDX, colIDX);
					unsigned char LBP_bNum = 0;
					//create the binary number from the bool PX > center. This is the LBP
					LBP_bNum |= (image.at<uchar>(rowIDX - 1, colIDX - 1) > centerPX) << 7;
					LBP_bNum |= (image.at<uchar>(rowIDX - 1, colIDX    ) > centerPX) << 6;
					LBP_bNum |= (image.at<uchar>(rowIDX - 1, colIDX + 1) > centerPX) << 5;
					LBP_bNum |= (image.at<uchar>(rowIDX    , colIDX + 1) > centerPX) << 4;
					LBP_bNum |= (image.at<uchar>(rowIDX + 1, colIDX + 1) > centerPX) << 3;
					LBP_bNum |= (image.at<uchar>(rowIDX + 1, colIDX    ) > centerPX) << 2;
					LBP_bNum |= (image.at<uchar>(rowIDX + 1, colIDX - 1) > centerPX) << 1;
					LBP_bNum |= (image.at<uchar>(rowIDX    , colIDX - 1) > centerPX) << 0;
					*/
				}
			}
		}
		if (numNeighbors == 16)
		{
			//????????????? are we using interpolation?
		}
	}
	if (radius == 2)
	{
		if (numNeighbors == 8)
		{
			
			//copy apste starts here
			
			//the output amtrix will have a reduces size by 2 on each edge
			Mat LBP_mat = Mat::zeros(image.rows - 4, image.cols - 4, CV_8UC1);
			
			for (int rowIDX = 2; rowIDX < (image.rows - 2); rowIDX++)
			{
				for (int colIDX = 2; colIDX < (image.cols - 2); colIDX++)
				{
					//WHAT IS THE PIXEL TYPE FROM TEH IMAGES??
					int centerPX = image.at<int>(rowIDX, colIDX);

					//starting from the top left corner, c is center pixel, 0,0
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


					//moving clockwise, 1 if the square is > center pixel value.
					int LBP_1 = (image.at<uchar>(rowIDX - 1, colIDX - 1) > centerPX) ? 1 : 0;
					int LBP_2 = (image.at<uchar>(rowIDX - 1, colIDX)     > centerPX) ? 1 : 0;
					int LBP_3 = (image.at<uchar>(rowIDX - 1, colIDX + 1) > centerPX) ? 1 : 0;
					int LBP_4 = (image.at<uchar>(rowIDX, colIDX + 1)     > centerPX) ? 1 : 0;
					int LBP_5 = (image.at<uchar>(rowIDX + 1, colIDX + 1) > centerPX) ? 1 : 0;
					int LBP_6 = (image.at<uchar>(rowIDX + 1, colIDX)     > centerPX) ? 1 : 0;
					int LBP_7 = (image.at<uchar>(rowIDX + 1, colIDX - 1) > centerPX) ? 1 : 0;
					int LBP_8 = (image.at<uchar>(rowIDX, colIDX - 1)     > centerPX) ? 1 : 0;

					//turn the binary digits into a decimal number
					int LBP = (LBP_1 * 2 ^ 7) + (LBP_2 * 2 ^ 6) + (LBP_3 * 2 ^ 5) + (LBP_4 * 2 ^ 4) + (LBP_5 * 2 ^ 3) + (LBP_6 * 2 ^ 2) + (LBP_7 * 2 ^ 1) + (LBP_8 * 2 ^ 0);

					//now check if it is a uniform LBP or not by counting the changes in bits. 
					//if more than two, then not uniform
					int binNum[8] = { LBP_1, LBP_2, LBP_3, LBP_4, LBP_5, LBP_6, LBP_7, LBP_8 };
					int changes = 0;
					for (int i = 0; i < (8 - 1); i++)
					{
						if (binNum[i] != binNum[i + 1])
						{
							changes++;
						}
					}
					//all non uniform numbers get assigned arbitrary number 10101010 = 170
					if (changes > 2)
					{
						int LBP = 170;
					}

					//set the output matrix at this pixel index as the value of the pixel LBP
					LBP_mat.at<uchar>(rowIDX - 1, colIDX - 1) = LBP; // -1 to ensure we start filling it up from (0,0)

		
					//A DIFFERENT METHOD WITH MORE BIT LOGIC
					uchar centerPX = image.at<uchar>(rowIDX, colIDX);
					unsigned char LBP_bNum = 0;
					//create the binary number from the bool PX > center. This is the LBP
					LBP_bNum |= (image.at<uchar>(rowIDX - 1, colIDX - 1) > centerPX) << 7;
					LBP_bNum |= (image.at<uchar>(rowIDX - 1, colIDX    ) > centerPX) << 6;
					LBP_bNum |= (image.at<uchar>(rowIDX - 1, colIDX + 1) > centerPX) << 5;
					LBP_bNum |= (image.at<uchar>(rowIDX    , colIDX + 1) > centerPX) << 4;
					LBP_bNum |= (image.at<uchar>(rowIDX + 1, colIDX + 1) > centerPX) << 3;
					LBP_bNum |= (image.at<uchar>(rowIDX + 1, colIDX    ) > centerPX) << 2;
					LBP_bNum |= (image.at<uchar>(rowIDX + 1, colIDX - 1) > centerPX) << 1;
					LBP_bNum |= (image.at<uchar>(rowIDX    , colIDX - 1) > centerPX) << 0;
		
				}
			}
			
			//copy paste ends here
			
		}
		if (numNeighbors == 16)
		{

			/*

			Mat LBP_mat = Mat::zeros(image.rows - 2, image.cols - 2, CV_8UC1);
			
			for (int rowIDX = 1; rowIDX < (image.rows - 1); rowIDX++)
			{
				for (int colIDX = 1; colIDX < (image.cols - 1); colIDX++)
				{
					//WHAT IS THE PIXEL TYPE FROM TEH IMAGES??
					int centerPX = image.at< //CHECK THISint>(rowIDX, colIDX);

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
					int LBP_1 = (image.at<uchar>(rowIDX - 1, colIDX - 1) > centerPX) ? 1 : 0;
					int LBP_2 = (image.at<uchar>(rowIDX - 1, colIDX)     > centerPX) ? 1 : 0;
					int LBP_3 = (image.at<uchar>(rowIDX - 1, colIDX + 1) > centerPX) ? 1 : 0;
					int LBP_4 = (image.at<uchar>(rowIDX, colIDX + 1)     > centerPX) ? 1 : 0;
					int LBP_5 = (image.at<uchar>(rowIDX + 1, colIDX + 1) > centerPX) ? 1 : 0;
					int LBP_6 = (image.at<uchar>(rowIDX + 1, colIDX)     > centerPX) ? 1 : 0;
					int LBP_7 = (image.at<uchar>(rowIDX + 1, colIDX - 1) > centerPX) ? 1 : 0;
					int LBP_8 = (image.at<uchar>(rowIDX, colIDX - 1)     > centerPX) ? 1 : 0;

					//turn the binary digits into a decimal number
					int LBP = (LBP_1 * 2 ^ 7) + (LBP_2 * 2 ^ 6) + (LBP_3 * 2 ^ 5) + (LBP_4 * 2 ^ 4) + (LBP_5 * 2 ^ 3) + (LBP_6 * 2 ^ 2) + (LBP_7 * 2 ^ 1) + (LBP_8 * 2 ^ 0);

					//now check if it is a uniform LBP or not by counting the changes in bits.
					//if more than two, then not uniform
					int binNum[8] = { LBP_1, LBP_2, LBP_3, LBP_4, LBP_5, LBP_6, LBP_7, LBP_8 };
					int changes = 0;
					for (int i = 0; i < (8 - 1); i++)
					{
					if (binNum[i] != binNum[i + 1])
					{
					changes++;
					}
					}
					//all non uniform numbers get assigned arbitrary number 10101010 = 170
					if (changes > 2)
					{
					int LBP = 170;
					}

					//set the output matrix at this pixel index as the value of the pixel LBP
					LBP_mat.at<uchar>(rowIDX - 1, colIDX - 1) = LBP; // -1 to ensure we start filling it up from (0,0)


					//A DIFFERENT METHOD WITH MORE BIT LOGIC
					uchar centerPX = image.at<uchar>(rowIDX, colIDX);
					unsigned char LBP_bNum = 0;
					//create the binary number from the bool PX > center. This is the LBP
					LBP_bNum |= (image.at<uchar>(rowIDX - 1, colIDX - 1) > centerPX) << 7;
					LBP_bNum |= (image.at<uchar>(rowIDX - 1, colIDX    ) > centerPX) << 6;
					LBP_bNum |= (image.at<uchar>(rowIDX - 1, colIDX + 1) > centerPX) << 5;
					LBP_bNum |= (image.at<uchar>(rowIDX    , colIDX + 1) > centerPX) << 4;
					LBP_bNum |= (image.at<uchar>(rowIDX + 1, colIDX + 1) > centerPX) << 3;
					LBP_bNum |= (image.at<uchar>(rowIDX + 1, colIDX    ) > centerPX) << 2;
					LBP_bNum |= (image.at<uchar>(rowIDX + 1, colIDX - 1) > centerPX) << 1;
					LBP_bNum |= (image.at<uchar>(rowIDX    , colIDX - 1) > centerPX) << 0;

				}
			}

			*/

		}
	}
}