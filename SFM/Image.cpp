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
	return Image::loadFF(path);
}

vector<Mat> Image::loadFF(string path){

	#if defined VERBOSE
	cout << "Searching for images" << endl;
	#endif	
	// std::vector of Mats
	vector<Mat> images;

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

	// Folder traversal
	DIR *dir;
	struct dirent *ent;

	if ((dir = opendir (path.c_str())) != NULL) {
	  
	  	while ((ent = readdir (dir)) != NULL) {
			// If name equals regex, load into cv::Mat and append to images
			if(match (ent->d_name, paths)){
				realPath = path + "/" + (ent->d_name);
				#if defined VERBOSE
				cout << (ent->d_name) << endl;
				#endif
				images.push_back(imread( realPath , cv::IMREAD_COLOR));
			}
	  	}
	
	  	#if defined DEBUG || defined VERBOSE
	  	cout << "Loaded " << images.size() << " images" << endl;
	  	#endif

		closedir (dir);
	} else {
		perror ("");
	}
	return images;
}
