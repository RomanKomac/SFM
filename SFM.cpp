#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "SFM/Image.hpp"
#include "SFM/RANSAC/ransac.hpp"

using namespace cv;
using namespace std;

void SFMPipeline(string path);
bool sortMatches(DMatch m1, DMatch m2);

int main(int argc, char** argv )
{
	if (argc >= 2){
    	SFMPipeline(argv[1]);

	} else {
		cout << "USAGE" << endl;
		cout << "./StructureFromMotion arg1 arg2 ..." << endl;
		cout << "arg1 must be a directory that contains ImageList.txt" << endl;
		cout << "ImageList.txt should contain lines of relative image paths." << endl;
		cout << "Other arguments are currently optional" << endl;
	}
	

    return 0;
}

bool sortMatches(DMatch m1, DMatch m2){
	return m1.distance < m2.distance;
}

void SFMPipeline(string path){
	// Loads images into an array
	vector<Mat> imgs;
	imgs = Image::loadFromFolder(string(path));


	#if defined VERBOSE
	cout << "Converting images to grayscale" << endl;
	#endif	
	
	// Convert images to grayscale
	for(int i = 0; i < imgs.size(); i++){
		Mat greyMat;
		cvtColor(imgs[i], greyMat, COLOR_BGR2GRAY);
		imgs[i] = greyMat;
	}

	#if defined VERBOSE
	cout << "Extracting keypoints" << endl;
	#endif	
	// Finds and extracts keypoints
	vector< vector< KeyPoint > > keypoints;
	vector< Mat > descriptors;
	
	SiftFeatureDetector detector;
	SurfDescriptorExtractor extractor;

	for(int i = 0; i < imgs.size(); i++){
		vector< KeyPoint > kpts;
		Mat desc;
		detector.detect(imgs[i], kpts);
		extractor.compute(imgs[i], kpts, desc);
		keypoints.push_back(kpts);
		descriptors.push_back(desc);
	}


	#if defined VERBOSE
	cout << "Matching descriptors" << endl;
	#endif	
	// Matches descriptors of image pairs
	// In this step we assume consecutiveness of pairs
	FlannBasedMatcher matcher;
	vector< vector< DMatch > > matches;
	for(int i = 1; i < imgs.size(); i++){
		vector< DMatch > localMatches;
		matcher.match(descriptors[i-1], descriptors[i], localMatches);
		matches.push_back(localMatches);
	}

	#if defined VERBOSE
	cout << "Sorting matches" << endl;
	#endif	
	// Sorting matches by match strength
	for(int i = 0; i < matches.size(); i++){
		sort(matches[i].begin(), matches[i].end(), sortMatches);
	}

	

}