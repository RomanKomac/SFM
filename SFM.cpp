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
	cout << "Matching and filtering descriptors" << endl;
	#endif	
	// Matches descriptors of image pairs
	// In this step we assume consecutiveness of pairs
	BFMatcher matcher;
	vector< vector< DMatch > > matches;

	// Rule of thumb proposed in Lowe's paper
	// If the (strength_first_match < ratio*strength_second_match) then keep
	const float ratio = 0.8;

	for(int i = 1; i < imgs.size(); i++){
		vector< vector< DMatch > > localMatches;
		vector< DMatch > goodMatches;
		matcher.knnMatch(descriptors[i-1], descriptors[i], localMatches, 2);
		for (int j = 0; j < localMatches.size(); j++)
		{
		    if (localMatches[j][0].distance < ratio * localMatches[j][1].distance)
		    {
		        goodMatches.push_back(localMatches[j][0]);
		    }
		}
		matches.push_back(goodMatches);
	}



	vector<Point2f> points1;
	vector<Point2f> points2;
	vector<float> similarities;
	for(int k = 0; k < matches[0].size(); k++){
		points1.push_back(keypoints[0][matches[0][k].queryIdx].pt);
		points2.push_back(keypoints[1][matches[0][k].trainIdx].pt);
		similarities.push_back(matches[0][k].distance);
	}


	// Reprojection error, the tolerance of the estimated inlier model
	double reprError = 1;
	// COnfidence of the model. 0.99 means 99% probability of it being correct.
	double confidence = 0.99;
	Mat mask1;
	Mat mask2;
	Mat F;
	//Mat fund_mat = findFundamentalMat(points1, points2, FM_RANSAC, reprError, confidence, mask1);

	//cout << fund_mat << endl;
	//cout << F << endl;

	F = Estimator::estFundamentalMat(points1, points2, SFM_RANSAC, reprError, confidence, mask2, similarities);
	cout << F << endl;

	//int numm = countNonZero(mask2);
	vector<Point2f> pointsbet1;
	vector<Point2f> pointsbet2;

	vector<DMatch> good_matches;
	for(int l = 0; l < matches[0].size(); l++){
		if(mask2.at<unsigned char>(l)){
			good_matches.push_back(matches[0][l]);
			pointsbet1.push_back(points1[l]);
			pointsbet2.push_back(points2[l]);
		}
	}



	Mat img_matches;
	drawMatches( imgs[0], keypoints[0], imgs[1], keypoints[1],
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


	Size size(img_matches.cols/2, img_matches.rows/2);
	Mat dst;//dst image
	resize(img_matches,dst,size);
  	//-- Show detected matches
  	imshow( "Good Matches", dst );
  	waitKey(-1);
  	/*
	#if defined VERBOSE
	cout << "Estimating fundamental matrices" << endl;
	#endif	
	// Extracting points by match strength
	for(int i = 0; i < matches.size(); i++){
		Mat points1(matches[i].size(), 3, CV_64FC1);
		Mat points2(matches[i].size(), 3, CV_64FC1);
		// Sets third column to 1, homogenous coordinate system
		points1.col(2) = 1;
		points2.col(2) = 1;
		// Fill others with point coordinates
		for(int j = 0; j < matches[i].size()-1; j++){
			KeyPoint kp1 = keypoints[i][matches[i][j].queryIdx];
			KeyPoint kp2 = keypoints[i][matches[i][j].queryIdx];
			points1.at<double>(j,0) = 
			points1.at<double>(j,1) = matches[i][j].queryIdx
		}
		
	}


	
	#if defined VERBOSE
	cout << "" << endl;
	#endif
	*/
}