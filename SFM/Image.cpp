#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <dirent.h>
#include <fstream>
#include "Image.hpp"

using namespace std;
using namespace cv;


bool match(string fname, vector<string> paths) {
	for(int i = 0; i < paths.size(); i++){
		if(paths[i] == fname)
			return true;
	}
	return false;
}


vector<Mat> Image::loadFromFolder(string path){
	return Image::loadFF(path, "");
}

vector<Mat> Image::loadFromFolder(string path, string pattern){
	return Image::loadFF(path, pattern);
}


vector<Mat> Image::loadFF(string path, string pattern = ""){

	#if defined VERBOSE
	cout << "Searching for images" << endl;
	#endif	

	if(pattern == string("")){
		string realPath;
		// Path to the file with image names
		string descriptionFilePath = path + "/ImageList.txt";
		vector<string> paths;

		ifstream descFile;
		descFile.open(descriptionFilePath.c_str());
		string line;
		while(getline(descFile, line)){
			paths.push_back(line);
		}
		descFile.close();

		// std::vector of Mats
		vector<Mat> images(paths.size());

		// Folder traversal
		DIR *dir;
		struct dirent *ent;

		if ((dir = opendir (path.c_str())) != NULL) {
		  
			for(int i = 0; i < paths.size(); i++){
				
				realPath = path + "/" + paths[i];
				#if defined VERBOSE
				cout << realPath << endl;
				#endif
				images[i] = (imread( realPath , cv::IMREAD_COLOR));
			}

		  	#if defined DEBUG || defined VERBOSE
		  	cout << "Loaded " << images.size() << " images" << endl;
		  	#endif

			closedir (dir);
		} else {
			perror ("");
		}
		return images;
	} else {

		if(path != "" && (path.at(path.size()-1) != '/' || path.at(path.size()-1) != '\\'))
			path += "/";

		VideoCapture cap(path+pattern);

		if(!cap.isOpened()) {
			cout << "Failed to open image sequence" << endl;
			return vector<Mat>();
		}

		vector<Mat> images;

		Mat img;
		while(cap.read(img)){
			images.push_back(img.clone());
		}

		#if defined DEBUG || defined VERBOSE
	  	cout << "Loaded " << images.size() << " images" << endl;
	  	#endif

		return images;
	}
}
