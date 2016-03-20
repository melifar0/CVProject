#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace cv;
using namespace std;

class ImageDataSet
{
public:ImageDataSet::ImageDataSet(string QMULPath, string HeadPoseDBPath)
	{
		cout << "Loading QMUL Multiview Face Dataset" << endl;
		QMUL_Images = vector<vector<vector<Mat>>>(QMUL_SubjectIDs.size());
		
		for (int i = 0; i < QMUL_SubjectIDs.size(); i++) {
			string subjectPath = QMULPath + "/" + QMUL_SubjectIDs[i] + "Grey";
			for (int j = 0; j < QMUL_TiltCodes.size(); j++) {
				vector<Mat> tmp;
				for (int k = 0; k < QMUL_PanCodes.size(); k++) {
					string imageFilename = QMUL_SubjectIDs[i] + "_" + QMUL_TiltCodes[j] + "_" + QMUL_PanCodes[k] + ".ras";
					string imageFilepath = subjectPath + "/" + imageFilename;
					
					// Load image
					Mat img = imread(imageFilepath, CV_LOAD_IMAGE_COLOR);
					
					// Check image data
					if (!img.data) {
						cout << "\t\tError loading image " << imageFilepath << endl;
						return;
					}
					
					// Save image to the data set
					tmp.push_back(img);
				}
				QMUL_Images[i].push_back(tmp);
			}
			
		}
		cout << "QMUL Multiview Face Dataset Loaded Successfully" << endl;
	
		cout << "Loading Head Pose Image Database" << endl;

	}

	//Get all images of a given subject
	vector<vector<Mat>> QMUL_getImagesBySubjectID(string SubjectID) 
	{
		int subjPos = find(QMUL_SubjectIDs.begin(), QMUL_SubjectIDs.end(), SubjectID) - QMUL_SubjectIDs.begin();
		if (subjPos >= QMUL_SubjectIDs.size()){
			cout << "ERROR in QMUL_getImagesBySubjectID" << endl;
			cout << SubjectID << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
			
		}

		return QMUL_Images[subjPos];
	}

	//Get the image of a tilt-pan combination of a given subject
	Mat QMUL_getSubjectImageByPose(string SubjectID, string TiltCode, string PanCode)
	{
		int subjPos = find(QMUL_SubjectIDs.begin(), QMUL_SubjectIDs.end(), SubjectID) - QMUL_SubjectIDs.begin();
		if (subjPos >= QMUL_SubjectIDs.size()){
			cout << "ERROR in QMUL_getSubjectImageByPose" << endl;
			cout << SubjectID << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}

		int tiltPos = find(QMUL_TiltCodes.begin(), QMUL_TiltCodes.end(), TiltCode) - QMUL_TiltCodes.begin();
		if (tiltPos >= QMUL_TiltCodes.size()){
			cout << "ERROR in QMUL_getSubjectImageByPose" << endl;
			cout << TiltCode << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}

		int PanPos = find(QMUL_PanCodes.begin(), QMUL_PanCodes.end(), PanCode) - QMUL_PanCodes.begin();
		if (PanPos >= QMUL_PanCodes.size()){
			cout << "ERROR in QMUL_getSubjectImageByPose" << endl;
			cout << PanCode << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}
		
		return QMUL_Images[subjPos][tiltPos][PanPos];
	}

	//Get all images of a given tilt-pan combination for all subjects
	vector<Mat> QMUL_getImagesByPose(string TiltCode, string PanCode) 
	{
		vector<Mat> images;

		for (int i = 0; i < QMUL_SubjectIDs.size(); i++) {
			images.push_back(QMUL_getSubjectImageByPose(QMUL_SubjectIDs[i], TiltCode, PanCode));
		}

		return images;
	}

	//Display all images of a given subject
	void QMUL_displaySubjectImages(string SubjectID) {
		vector<vector<Mat>> images = QMUL_getImagesBySubjectID(SubjectID);
		int width = 80 * QMUL_PanCodes.size();
		int height = 80 * QMUL_TiltCodes.size();
				
		vector<Mat> mergedH(images.size());
		for (int i = 0; i < images.size(); i++) {
			hconcat(images[i], mergedH[i]);
		}

		Mat mergedV;
		vconcat(mergedH, mergedV);

		Mat resized;
		resize(mergedV, resized, Size(width, height));
		imshow("Resized images", resized);
		waitKey(0);
	}

	bool isSuccessfullyLoaded() { return successfullyLoaded; }

	vector<vector<vector<Mat>>> QMUL_Images;

	const vector<string> QMUL_SubjectIDs = vector<string>
	({
		"AdamB","AndreeaV","CarlaB","ColinP","DanJ",
		"DennisP","DennisPNoGlasses","DerekC","GrahamW","HeatherL",
		"Jack","JamieS","JeffNG","John","OngEJ",
		"KateS","KatherineW","KeithC","KrystynaN","PaulV",
		"RichardB","RichardH","SarahL","SeanG","SeanGNoGlasses"
		,"SimonB","SueW","TasosH","TomK","YogeshR","YongminY"
	});
	

	const vector<string> QMUL_TiltCodes = vector<string>
	({
		"060", "070", "080", "090", "100", "110", "120"
	});

	const vector<string> QMUL_PanCodes = vector<string>
	({
		"000", "010", "020", "030", "040", "050", "060", "070", "080", "090",
		"100", "110", "120", "130", "140", "150", "160", "170", "180"
	});

private:
	bool successfullyLoaded = false;
};


void main(void)
{
	/* Initialize OpenCV nonfree module */
	initModule_nonfree();

	/* Put the full path of the QMUL Multiview Face Dataset folder here */
	const string QMULPath = "C:/Users/Salian/Dropbox/Winter2016/Vision/CVProject/CVProject/resources/QMUL_360degreeViewSphere_FaceDatabase/Set1_Greyscale";

	/* Put the full path of the Head Pose Image Database folder here */
	const string HeadPoseDBPath = "C:/Users/Salian/Dropbox/Winter2016/Vision/CVProject/CVProject/resources/HeadPoseImageDatabase";

	/* Load the dataset by instantiating the helper class */
	ImageDataSet data(QMULPath, HeadPoseDBPath);

	/* Terminate if data is not successfull loaded */
	if (!data.isSuccessfullyLoaded())
	{
		cout << "An error occurred, press Enter to exit" << endl;
		getchar();
		return;
	}
}