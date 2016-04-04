#ifndef IMAGEDATASET_H
#define IMAGEDATASET_H

#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

class ImageDataSet
{
private:
	bool successfullyLoaded = false;

public:
	ImageDataSet(string QMULPath, string HPDBPath);

	vector<vector<Mat>> QMUL_getImagesBySubjectID(string SubjectID);
	Mat QMUL_getSubjectImageByPose(string SubjectID, string TiltCode, string PanCode);
	vector<Mat> QMUL_getImagesByPose(string TiltCode, string PanCode);
	void QMUL_displaySubjectImages(string SubjectID);

	vector<vector<Mat>> HPDB_getImagesBySubjectIDandSerie(string SubjectID, string Serie);
	vector<vector<Rect>> HPDB_getLabelsBySubjectIDandSerie(string SubjectID, string Serie);
	Mat HPDB_getSubjectImageByPoseAndSerie(string SubjectID, string Serie, string TiltCode, string PanCode);
	vector<Mat> HPDB_getImagesByPose(string TiltCode, string PanCode);
	void HPDB_displayImagesBySubjectIDandSerie(string SubjectID, string Serie);

	bool isSuccessfullyLoaded();

	vector<vector<vector<Mat>>> QMUL_Images;

	const vector<string> QMUL_SubjectIDs = vector<string>
		({
		"AdamB", "AndreeaV", "CarlaB", "ColinP", "DanJ",
		"DennisP", "DennisPNoGlasses", "DerekC", "GrahamW", "HeatherL",
		"Jack", "JamieS", "JeffNG", "John", "OngEJ",
		"KateS", "KatherineW", "KeithC", "KrystynaN", "PaulV",
		"RichardB", "RichardH", "SarahL", "SeanG", "SeanGNoGlasses"
		, "SimonB", "SueW", "TasosH", "TomK", "YogeshR", "YongminY"
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

};

#endif