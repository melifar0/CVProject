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
	//imshow("Test1", data.QMUL_getSubjectImageByPose("AdamB", "090", "090"));
	//waitKey(0);
	//destroyWindow("Test1");
	//imshow("Test2", data.HPDB_getSubjectImageByPoseAndSerie("Person01", "1", "-15", "+0"));
	//waitKey(0);
	//destroyWindow("Test2");

	//implement the rest of the code here
	EigenFaces eigen;
	eigen.crossValidation(data);
}

