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
public:ImageDataSet::ImageDataSet(string QMULPath, string HPDBPath)
	{
		cout << "Loading QMUL Multiview Face Dataset" << endl;
		//QMUL_Images are a 3D matrix - QMUL_Images[SubjectID][Tilt][Pan]
		QMUL_Images = vector<vector<vector<Mat>>>(QMUL_SubjectIDs.size());
		
		for (int i = 0; i < QMUL_SubjectIDs.size(); i++) {
			string subjectPath = QMULPath + "/" + QMUL_SubjectIDs[i] + "Grey";
			for (int j = 0; j < QMUL_TiltCodes.size(); j++) {
				vector<Mat> tmp;
				for (int k = 0; k < QMUL_PanCodes.size(); k++) {
					string imageFilename = QMUL_SubjectIDs[i] + "_" + QMUL_TiltCodes[j] + "_" + QMUL_PanCodes[k] + ".ras";
					string imageFilepath = subjectPath + "/" + imageFilename;
					
					//Load image
					Mat img = imread(imageFilepath, CV_LOAD_IMAGE_COLOR);
					
					//Check image data
					if (!img.data) {
						cout << "\t\tError loading image " << imageFilepath << endl;
						return;
					}
					
					//Save image to the data set
					tmp.push_back(img);
				}
				QMUL_Images[i].push_back(tmp);
			}
			
		}
		cout << "QMUL Multiview Face Dataset Loaded Successfully" << endl;
	
		cout << "Loading Head Pose Image Database" << endl;
		//HPDB_Images are a 3D matrix - HPDB_Images[SubjectIDs][Tilt][Pan]
		HPDB_Images = vector<vector<vector<Mat>>>(HPDB_SubjectIDs.size());
		HPDB_Labels = vector<vector<vector<Rect>>>(HPDB_SubjectIDs.size());
		for (int i = 0; i < HPDB_SubjectIDs.size(); i++) {
			string personPath = HPDBPath + "/" + HPDB_SubjectIDs[i];
			for (int n = 0; n < HPDB_Series.size(); n++) {
				for (int j = 0; j < HPDB_TiltCodes.size(); j++) {
					vector<Mat> tmp;
					vector<Rect> rect_tmp;
					for (int k = 0; k < HPDB_PanCodes.size(); k++) {
						string imageFilepath = personPath + "/" + HPDB_SubjectIDs[i] + HPDB_Series[n] + to_string(14 + j*HPDB_PanCodes.size() + k) + HPDB_TiltCodes[j] + HPDB_PanCodes[k] + ".jpg";
						string labelFilepath = personPath + "/" + HPDB_SubjectIDs[i] + HPDB_Series[n] + to_string(14 + j*HPDB_PanCodes.size() + k) + HPDB_TiltCodes[j] + HPDB_PanCodes[k] + ".txt";
						
						//Load annotations
						ifstream infile(labelFilepath);

						//Checking annotation file
						if (!infile.is_open())
						{
							cout << "\t\tError: Error loading annotation in " << labelFilepath << endl;
							return;
						}

						bool parse = false;
						vector<int> coords;
						for (string line; getline(infile, line);)
						{
							if (parse) {
								coords.push_back(stoi(line));
							}
							if (line == "Face") {
								parse = true;
							}
						}

						//Make sure all coordinates got parsed
						if (!(coords.size() == 4)) {
							cout << "\t\tError: Error parsing annotation in " << labelFilepath << endl;
							return;
						}
						//Save label to the data set
						Rect labelRect = Rect(coords[0] - 51, coords[1] - 51, 100, 100);
						rect_tmp.push_back(labelRect);
						//Load image
						Mat img = imread(imageFilepath, CV_LOAD_IMAGE_COLOR);
						//Check image data
						if (!img.data) {
							cout << "\t\tError loading image " << imageFilepath << endl;
							return;
						}
						//Save image to the data set
						tmp.push_back(img);
					}
					HPDB_Images[i].push_back(tmp);
					HPDB_Labels[i].push_back(rect_tmp);
				}
			}
		}

		successfullyLoaded = true;
	}

	//Get all images of a given subject. Returns a 2D matrix where rows are tilt angles and columns are pan angles
	vector<vector<Mat>> QMUL_getImagesBySubjectID(string SubjectID) 
	{
		int subjPos = find(QMUL_SubjectIDs.begin(), QMUL_SubjectIDs.end(), SubjectID) - QMUL_SubjectIDs.begin();
		if (subjPos >= QMUL_SubjectIDs.size()) {
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
		if (subjPos >= QMUL_SubjectIDs.size()) {
			cout << "ERROR in QMUL_getSubjectImageByPose" << endl;
			cout << SubjectID << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}

		int tiltPos = find(QMUL_TiltCodes.begin(), QMUL_TiltCodes.end(), TiltCode) - QMUL_TiltCodes.begin();
		if (tiltPos >= QMUL_TiltCodes.size()) {
			cout << "ERROR in QMUL_getSubjectImageByPose" << endl;
			cout << TiltCode << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}

		int PanPos = find(QMUL_PanCodes.begin(), QMUL_PanCodes.end(), PanCode) - QMUL_PanCodes.begin();
		if (PanPos >= QMUL_PanCodes.size()) {
			cout << "ERROR in QMUL_getSubjectImageByPose" << endl;
			cout << PanCode << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}
		
		return QMUL_Images[subjPos][tiltPos][PanPos];
	}

	//Get all images of a given tilt-pan combination for all subjects.
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
		imshow(SubjectID + " images", resized);
		waitKey(0);
		destroyWindow(SubjectID + " images");
	}

	//Get all images of a given subject and serie number. Returns a 2D matrix where rows are tilt angles and columns are pan angles
	vector<vector<Mat>> HPDB_getImagesBySubjectIDandSerie(string SubjectID, string Serie)
	{
		vector<vector<Mat>> images;
		int tiltPerSerie = HPDB_TiltCodes.size();
		int subjPos = find(HPDB_SubjectIDs.begin(), HPDB_SubjectIDs.end(), SubjectID) - HPDB_SubjectIDs.begin();
		if (subjPos >= HPDB_SubjectIDs.size()) {
			cout << "ERROR in HPDB_getImagesBySubjectIDandSerie" << endl;
			cout << SubjectID << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}
		if (Serie == "1") {
			for (int i = 0; i < tiltPerSerie; i++) {
				images.push_back(HPDB_Images[subjPos][i]);
			}
		}
		else if (Serie == "2") {
			for (int i = 0; i < tiltPerSerie; i++) {
				images.push_back(HPDB_Images[subjPos][i + tiltPerSerie]);
			}
		}
		else {
			cout << "ERROR in HPDB_getImagesBySubjectIDandSerie" << endl;
			cout << Serie << " not valid." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}
		return images;
	}

	//Get all labels of a given subject and serie number. Returns a 2D matrix where rows are tilt angles and columns are pan angles
	vector<vector<Rect>> HPDB_getLabelsBySubjectIDandSerie(string SubjectID, string Serie)
	{
		vector<vector<Rect>> labels;
		int tiltPerSerie = HPDB_TiltCodes.size();
		int subjPos = find(HPDB_SubjectIDs.begin(), HPDB_SubjectIDs.end(), SubjectID) - HPDB_SubjectIDs.begin();
		if (subjPos >= HPDB_SubjectIDs.size()) {
			cout << "ERROR in HPDB_getLabelsBySubjectIDandSerie" << endl;
			cout << SubjectID << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}
		if (Serie == "1") {
			for (int i = 0; i < tiltPerSerie; i++) {
				labels.push_back(HPDB_Labels[subjPos][i]);
			}
		}
		else if (Serie == "2") {
			for (int i = 0; i < tiltPerSerie; i++) {
				labels.push_back(HPDB_Labels[subjPos][i + tiltPerSerie]);
			}
		}
		else {
			cout << "ERROR in HPDB_getLabelsBySubjectIDandSerie" << endl;
			cout << Serie << " not valid." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}
		return labels;
	}

	//Get the image of a tilt-pan-serie combination of a given subject
	Mat HPDB_getSubjectImageByPoseAndSerie(string SubjectID, string Serie, string TiltCode, string PanCode)
	{
		int subjPos = find(HPDB_SubjectIDs.begin(), HPDB_SubjectIDs.end(), SubjectID) - HPDB_SubjectIDs.begin();
		if (subjPos >= HPDB_SubjectIDs.size()) {
			cout << "ERROR in HPDB_getSubjectImageByPose" << endl;
			cout << SubjectID << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}

		int tiltPos = find(HPDB_TiltCodes.begin(), HPDB_TiltCodes.end(), TiltCode) - HPDB_TiltCodes.begin();
		if (tiltPos >= HPDB_TiltCodes.size()) {
			cout << "ERROR in HPDB_getSubjectImageByPose" << endl;
			cout << TiltCode << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}

		int PanPos = find(HPDB_PanCodes.begin(), HPDB_PanCodes.end(), PanCode) - HPDB_PanCodes.begin();
		if (PanPos >= HPDB_PanCodes.size()) {
			cout << "ERROR in HPDB_getSubjectImageByPose" << endl;
			cout << PanCode << " not found." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}

		if (!(Serie == "1" || Serie == "2")) {
			cout << "ERROR in HPDB_getSubjectImageByPoseSerie" << endl;
			cout << Serie << " not valid." << endl;
			cout << "The program will now exit, press Enter key" << endl;
			getchar();
			exit(1);
		}

		return HPDB_Images[subjPos][tiltPos+5*stoi(Serie)][PanPos];
	}

	//Get all images of a given tilt-pan combination for all subjects.
	vector<Mat> HPDB_getImagesByPose(string TiltCode, string PanCode)
	{
		vector<Mat> images;

		for (int i = 0; i < HPDB_SubjectIDs.size(); i++) {
			for (int j = 0; j < HPDB_Series.size(); j++) {
				images.push_back(HPDB_getSubjectImageByPoseAndSerie(HPDB_SubjectIDs[i], HPDB_Series[j], TiltCode, PanCode));
			}
		}

		return images;
	}

	//Display all images with labels of a given subject and serie number
	void HPDB_displayImagesBySubjectIDandSerie(string SubjectID, string Serie)
	{
		vector<vector<Mat>> images = HPDB_getImagesBySubjectIDandSerie(SubjectID, Serie);
		vector<vector<Rect>> labels = HPDB_getLabelsBySubjectIDandSerie(SubjectID, Serie);
		int width = 100 * HPDB_PanCodes.size();
		int height = 100 * HPDB_TiltCodes.size();

		for (int i = 0; i < images.size(); i++) {
			for (int j = 0; j < images[i].size(); j++) {
				rectangle(images[i][j], labels[i][j], Scalar(0, 0, 255), 6);
			}
		}

		vector<Mat> mergedH(images.size());
		for (int i = 0; i < images.size(); i++) {
			hconcat(images[i], mergedH[i]);
		}

		Mat mergedV;
		vconcat(mergedH, mergedV);

		Mat resized;
		resize(mergedV, resized, Size(width, height));
		imshow(SubjectID + " images", resized);
		waitKey(0);
		destroyWindow(SubjectID + " images");
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

	vector<vector<vector<Mat>>> HPDB_Images;
	vector<vector<vector<Rect>>> HPDB_Labels;

	const vector<string> HPDB_SubjectIDs = vector<string>
	({
		"Person01", "Person02", "Person03", "Person04", "Person05",
		"Person06", "Person07", "Person08", "Person09", "Person10",
		"Person11", "Person12", "Person13", "Person14", "Person15"
	});

	const vector<string> HPDB_Series = vector<string>
	({
		"1", "2"
	});

	const int HPDB_numberOfImagesPerSeries = 65;

	const vector<string> HPDB_TiltCodes = vector<string>
	({
		"-30", "-15", "+0", "+15", "+30"
	});

	const vector<string> HPDB_PanCodes = vector<string>
	({
		"-90", "-75", "-60", "-45", "-30", "-15", "+0", "+15", "+30", "+45", "+60", "+75", "+90"
	});

private:
	bool successfullyLoaded = false;
};


void main(void)
{
	// Initialize OpenCV nonfree module
	initModule_nonfree();

	//Put the full path of the QMUL Multiview Face Dataset folder here
	const string QMULPath = "C:/Users/Salian/Dropbox/Winter2016/Vision/CVProject/CVProject/resources/QMUL_360degreeViewSphere_FaceDatabase/Set1_Greyscale";

	//Put the full path of the Head Pose Image Database folder here
	const string HeadPoseDBPath = "C:/Users/Salian/Dropbox/Winter2016/Vision/CVProject/CVProject/resources/HeadPoseImageDatabase";

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
	imshow("Test1", data.QMUL_getSubjectImageByPose("AdamB", "090", "090"));
	waitKey(0);
	destroyWindow("Test1");
	imshow("Test2", data.HPDB_getSubjectImageByPoseAndSerie("Person01","1", "-15", "+0"));
	waitKey(0);
	destroyWindow("Test2");

	//implement the rest of the code here
}