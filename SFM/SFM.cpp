#include <opencv2/viz/vizcore.hpp>
#include "SFM.hpp"
#include "estimator/estimator.hpp"
#include "constants.hpp"

using namespace std;
using namespace cv;
using namespace viz;
using namespace xfeatures2d;

SFM::SFM(vector<Mat> images, Mat _K1, Mat _K2, vector<double> dist1, vector<double> dist2){
	BGR_imgs = images;
	GRAY_imgs = vector<Mat>();

	calibrated = true;
	K1 = _K1;
	K2 = _K2;
	distort1 = dist1;
	distort2 = dist2;

	for(int i = 0; i < images.size(); i++){
		Mat greyMat;
		cvtColor(images[i], greyMat, COLOR_BGR2GRAY);
		GRAY_imgs.push_back(greyMat);
	}
	mImageFeatures.resize(GRAY_imgs.size());
	mCameraPoses.resize(GRAY_imgs.size());
}

SFM::SFM(vector<Mat> images, Mat _K1, Mat _K2, vector<double> dist1){
	BGR_imgs = images;
	GRAY_imgs = vector<Mat>();

	calibrated = true;
	K1 = _K1;
	K2 = _K2;
	distort1 = dist1;
	distort2 = dist1;

	for(int i = 0; i < images.size(); i++){
		Mat greyMat;
		cvtColor(images[i], greyMat, COLOR_BGR2GRAY);
		GRAY_imgs.push_back(greyMat);
	}
	mImageFeatures.resize(GRAY_imgs.size());
	mCameraPoses.resize(GRAY_imgs.size());
}

SFM::SFM(vector<Mat> images, Mat _K1, Mat _K2){
	BGR_imgs = images;
	GRAY_imgs = vector<Mat>();

	calibrated = true;
	K1 = _K1;
	K2 = _K2;
	distort1 = vector<double>(5);
	distort2 = vector<double>(5);

	for(int i = 0; i < images.size(); i++){
		Mat greyMat;
		cvtColor(images[i], greyMat, COLOR_BGR2GRAY);
		GRAY_imgs.push_back(greyMat);
	}
	mImageFeatures.resize(GRAY_imgs.size());
	mCameraPoses.resize(GRAY_imgs.size());
}

SFM::SFM(vector<Mat> images, Mat _K1, vector<double> dist1){
	BGR_imgs = images;
	GRAY_imgs = vector<Mat>();

	calibrated = true;
	K1 = _K1;
	K2 = _K1;
	distort1 = dist1;
	distort2 = dist1;

	for(int i = 0; i < images.size(); i++){
		Mat greyMat;
		cvtColor(images[i], greyMat, COLOR_BGR2GRAY);
		GRAY_imgs.push_back(greyMat);
	}
	mImageFeatures.resize(GRAY_imgs.size());
	mCameraPoses.resize(GRAY_imgs.size());
}

SFM::SFM(vector<Mat> images, Mat _K1){
	BGR_imgs = images;
	GRAY_imgs = vector<Mat>();

	calibrated = true;
	K1 = _K1;
	K2 = _K1;
	distort1 = vector<double>(5);
	distort2 = vector<double>(5);

	for(int i = 0; i < images.size(); i++){
		Mat greyMat;
		cvtColor(images[i], greyMat, COLOR_BGR2GRAY);
		GRAY_imgs.push_back(greyMat);
	}
	mImageFeatures.resize(GRAY_imgs.size());
	mCameraPoses.resize(GRAY_imgs.size());
}

SFM::SFM(vector<Mat> images){
	BGR_imgs = images;
	GRAY_imgs = vector<Mat>();
	
	calibrated = false;
	distort1 = vector<double>(5);
	distort2 = vector<double>(5);

	for(int i = 0; i < images.size(); i++){
		Mat greyMat;
		cvtColor(images[i], greyMat, COLOR_BGR2GRAY);
		GRAY_imgs.push_back(greyMat);
	}
	mImageFeatures.resize(GRAY_imgs.size());
	mCameraPoses.resize(GRAY_imgs.size());
}

void SFM::detect(cv::Ptr<Feature2D> detector){
	for(int i = 0; i < GRAY_imgs.size(); i++){
		vector< KeyPoint > kpts;
		detector->detect(GRAY_imgs[i], kpts);
		keypoints.push_back(kpts);
		mImageFeatures[i].keyPoints = kpts;
		mImageFeatures[i].points.resize(kpts.size());
		for(int j = 0; j < kpts.size(); j++)
			mImageFeatures[i].points[j] = kpts[j].pt;
	}
}

void SFM::extract(cv::Ptr<Feature2D> extractor){
	for(int i = 0; i < GRAY_imgs.size(); i++){
		Mat desc;
		extractor->compute(GRAY_imgs[i], keypoints[i], desc);
		descriptors.push_back(desc);
	}
}

void SFM::match(DescriptorMatcher &matcher, int match_method = MATCH_CONSECUTIVE, int min_matches = 50){
	// Rule of thumb proposed in Lowe's paper
	// If the (strength_first_match < ratio*strength_second_match) then keep
	const float ratio = 0.8;
	matchMap = map< intPair,vector<DMatch> >();

	for(int i = 0; i < GRAY_imgs.size(); i++){
		int k = 0;
		if(match_method == MATCH_CONSECUTIVE){
			int s = GRAY_imgs.size();
			k = std::min(i+2,s);
		} else if(match_method == MATCH_EXHAUSTIVE){
			k = GRAY_imgs.size();
		}
		for(int j = i+1; j < k; j++){
			vector< vector< DMatch > > localMatches;
			vector< DMatch > goodMatches;
			matcher.knnMatch(descriptors[i], descriptors[j], localMatches, 2);
			for (int l = 0; l < localMatches.size(); l++)
			{
			    if (localMatches[l][0].distance < ratio * localMatches[l][1].distance)
			    {
			        goodMatches.push_back(localMatches[l][0]);
			    }
			}
			sort(goodMatches.begin(), goodMatches.end(), sortMatches); 
			matchMap.insert( pair< intPair, vector<DMatch> >(intPair(i,j), goodMatches) );
		}
	}
}

void SFM::RANSACfundamental(double reprError, double confidence, int method, int min_inliers = 50){

	avg_num_iters = 0;
	avg_runtime = 0;

	int cntr = 0;
	map< intPair, vector<DMatch> >::iterator it;
	for(it = matchMap.begin(); it != matchMap.end(); it++){
		vector<Point2f> points1;
		vector<Point2f> points2;
		vector<float> similarities;
		intPair pr = it->first;
		vector<DMatch> mch = it->second;

		for(int k = 0; k < mch.size(); k++){
			points1.push_back(keypoints[pr.first][mch[k].queryIdx].pt);
			points2.push_back(keypoints[pr.second][mch[k].trainIdx].pt);
			similarities.push_back(mch[k].distance);
		}
		
		Mat mask;
		Mat F;

		F = Estimator::estFundamentalMat(points1, points2, method, reprError, confidence, mask, similarities);
		if(countNonZero(mask) >= min_inliers){
			#if defined VERBOSE
			cout << "estimated fundamental matrix of pair " << pr.first << "|" << pr.second << endl;
			cout << F << endl;
			#endif
			fundMap.insert( pair< intPair, Mat >(pr,F) );
			maskMap.insert( pair< intPair, Mat >(pr,mask) );
		}
		
		avg_num_iters += Estimator::num_iters;
		avg_runtime += Estimator::runtime;
		cntr++;
	}
	avg_num_iters /= cntr;
	avg_runtime /= cntr;
}

void SFM::motionFromFundamental(){
	if(calibrated){

		map< intPair, vector<DMatch> >::iterator it;
		for(it = matchMap.begin(); it != matchMap.end(); it++){
			vector<Point2f> points1;
			vector<Point2f> points2;
			vector<float> similarities;
			intPair pr = it->first;
			vector<DMatch> mch = it->second;

			//Check if fundamental matrix was recovered for the pair
			map< intPair, Mat >::iterator fundIt;
			if( (fundIt = fundMap.find(pr)) != fundMap.end()){
				for(int k = 0; k < mch.size(); k++){
					points1.push_back(keypoints[pr.first][mch[k].queryIdx].pt);
					points2.push_back(keypoints[pr.second][mch[k].trainIdx].pt);
					similarities.push_back(mch[k].distance);
				}

				vector<Point2f> bestValidPoint1, bestValidPoint2;
				Mat mask = maskMap.find(pr)->second;
				for(int j = 0; j < mask.total(); j++){
					if(mask.at<uchar>(j)){
						bestValidPoint1.push_back(points1[j]);
						bestValidPoint2.push_back(points2[j]);
						break;
					}
				}

				Mat E, R, t, F = fundIt->second;
				E = Estimator::essentialFromFundamental(F, K1, K2);
				essenMap.insert( pair< intPair,Mat >(pr,E) );

				if(Estimator::motionFromEssential(E, K1, K2, bestValidPoint1, bestValidPoint2, R, t)){
					RtMap.insert( pair< intPair,MatPair >( pr,MatPair(R,t) ) );
					#if defined VERBOSE
					cout << "Recovered Rotation and Translation matrix for pair: " << pr.first << "|" << pr.second << endl;
					cout << R << endl;
					cout << t << endl;
					#endif	
				} else {
					#if defined VERBOSE
					cout << "Valid inter-camera motion could not be recovered for pair: " << pr.first << "|" << pr.second << endl;
					#endif
				}
			}
		}
	}
}

void SFM::initialBundleAdjustment(){
	sortedMasks = vector< pair< intPair,Mat > >();
	for(map< intPair, Mat >::iterator iter = maskMap.begin(); iter != maskMap.end(); iter++){
		sortedMasks.push_back(*iter);
	}
	sort(sortedMasks.begin(), sortedMasks.end(), sortMasks);

	for(int i = 0; i < sortedMasks.size(); i++){
		//If found Rotation and translation triangulate views

		intPair pr = sortedMasks[i].first;
		map< intPair, vector<DMatch> >::iterator mchIt;
		map< intPair, MatPair >::iterator RtIt;
		if( ((RtIt = RtMap.find(pr)) != RtMap.end()) && ((mchIt = matchMap.find(pr)) != matchMap.end()) ){
			Mat R = RtIt->second.first, t = RtIt->second.second;
			Matx34f Pleft = Matx34f::eye(), Pright;
    		Pright = Matx34f((float)R.at<double>(0,0), (float)R.at<double>(0,1), (float)R.at<double>(0,2), (float)t.at<double>(0),
			                 (float)R.at<double>(1,0), (float)R.at<double>(1,1), (float)R.at<double>(1,2), (float)t.at<double>(1),
			                 (float)R.at<double>(2,0), (float)R.at<double>(2,1), (float)R.at<double>(2,2), (float)t.at<double>(2));
			
    		vector< DMatch > mch = mchIt->second;
    		vector<Point2f> points1;
			vector<Point2f> points2;
			vector<intPair> indxs;

			PointCloud_f pointCloud;

			for(int k = 0; k < mch.size(); k++){
				points1.push_back(keypoints[pr.first][mch[k].queryIdx].pt);
				points2.push_back(keypoints[pr.second][mch[k].trainIdx].pt);
				indxs.push_back(intPair(mch[k].queryIdx, mch[k].trainIdx));
			}

			if(Estimator::triangulateViews(points1, points2, Pleft, Pright, indxs, pr, K1, pointCloud)){
				mReconstructionCloud = pointCloud;
				mCameraPoses[pr.first] = Pleft;
				mCameraPoses[pr.second] = Pright;
				mDoneViews.insert(pr.first);
				mDoneViews.insert(pr.second);
				mGoodViews.insert(pr.first);
				mGoodViews.insert(pr.second);
				ba.run(mReconstructionCloud, mCameraPoses, K1, mImageFeatures, BA_RE_SIMPLE);
				return;
			} else {
				//If points could not be triangulated from views
				continue;
			}
		} else {
			//Could not find associated Rotation, translation or match vector
			continue;
		}
	}

	cout << "Could not establish initial views" << endl;
}

void SFM::additionalBundleAdjustment(){
	for(int i = 0; i < sortedMasks.size(); i++){
		//If found Rotation and translation triangulate views

		intPair pr = sortedMasks[i].first;
		map< intPair, vector<DMatch> >::iterator mchIt;
		map< intPair, MatPair >::iterator RtIt;
		if( ((RtIt = RtMap.find(pr)) != RtMap.end()) && ((mchIt = matchMap.find(pr)) != matchMap.end()) ){
			Mat R = RtIt->second.first, t = RtIt->second.second;
			Matx34f Pleft = Matx34f::eye(), Pright;
    		Pright = Matx34f((float)R.at<double>(0,0), (float)R.at<double>(0,1), (float)R.at<double>(0,2), (float)t.at<double>(0),
			                 (float)R.at<double>(1,0), (float)R.at<double>(1,1), (float)R.at<double>(1,2), (float)t.at<double>(1),
			                 (float)R.at<double>(2,0), (float)R.at<double>(2,1), (float)R.at<double>(2,2), (float)t.at<double>(2));
			
    		vector< DMatch > mch = mchIt->second;
    		vector<Point2f> points1;
			vector<Point2f> points2;
			vector<intPair> indxs;

			PointCloud_f pointCloud;

			for(int k = 0; k < mch.size(); k++){
				points1.push_back(keypoints[pr.first][mch[k].queryIdx].pt);
				points2.push_back(keypoints[pr.second][mch[k].trainIdx].pt);
				indxs.push_back(intPair(mch[k].queryIdx, mch[k].trainIdx));
			}

			if(Estimator::triangulateViews(points1, points2, Pleft, Pright, indxs, pr, K1, pointCloud)){
				mergePointCloud(pointCloud);
				mCameraPoses[pr.first] = Pleft;
				mCameraPoses[pr.second] = Pright;
				mDoneViews.insert(pr.first);
				mDoneViews.insert(pr.second);
				mGoodViews.insert(pr.first);
				mGoodViews.insert(pr.second);
				continue;
			} else {
				//If points could not be triangulated from views
				continue;
			}
		} else {
			//Could not find associated Rotation, translation or match vector
			continue;
		}
	}
	ba.run(mReconstructionCloud, mCameraPoses, K1, mImageFeatures, BA_RE_SIMPLE);
}

void SFM::mergePointCloud(const PointCloud_f& cloud) {
    /*size_t newPoints = 0;
    size_t mergedPoints = 0;

    PointCloud_f::const_iterator it;
    for (it = cloud.begin(); it != cloud.end(); it++) {
        const Point3f newPoint = it->p; //new 3D point

        bool foundAnyMatchingExistingViews = false;
        bool foundMatching3DPoint = false;
        PointCloud_f::iterator it2;
        for (it2 = mReconstructionCloud.begin(); it2 != mReconstructionCloud.end(); it2++) {
            if (cv::norm(it2->p - newPoint) < MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE) {
                //This point is very close to an existing 3D cloud point
                foundMatching3DPoint = true;

                //Look for common 2D features to confirm match
                std::map<int, int>::const_iterator it3;
                for (it3 = it->views.begin(); it3 != it->views.end(); it3++) {
                    //kv.first = new point's originating view
                    //kv.second = new point's view 2D feature index

                	std::map<int, int>::const_iterator it4;
                    for (it4 = it2->views.begin(); it4 != it2->views.end(); it4++) {
                        //existingKv.first = existing point's originating view
                        //existingKv.second = existing point's view 2D feature index

                        bool foundMatchingFeature = false;

						const bool newIsLeft = it3->first < it4->first;
						const int leftViewIdx         = (newIsLeft) ? it3->first  : it4->first;
                        const int leftViewFeatureIdx  = (newIsLeft) ? it3->second : it4->second;
                        const int rightViewIdx        = (newIsLeft) ? it4->first  : it3->first;
                        const int rightViewFeatureIdx = (newIsLeft) ? it4->second : it3->second;

                        const Matching& matching = mFeatureMatchMatrix[leftViewIdx][rightViewIdx];
                        for (const DMatch& match : matching) {
                            if (    match.queryIdx == leftViewFeatureIdx
                                and match.trainIdx == rightViewFeatureIdx
                                and match.distance < MERGE_CLOUD_FEATURE_MIN_MATCH_DISTANCE) {

                            	mergeMatchMatrix[leftViewIdx][rightViewIdx].push_back(match);

                                //Found a 2D feature match for the two 3D points - merge
                                foundMatchingFeature = true;
                                break;
                            }
                        }

                        if (foundMatchingFeature) {
                            //Add the new originating view, and feature index
                            existingPoint.originatingViews[newKv.first] = newKv.second;

                            foundAnyMatchingExistingViews = true;

                        }
                    }
                }
            }
            if (foundAnyMatchingExistingViews) {
                mergedPoints++;
                break; //Stop looking for more matching cloud points
            }
        }

        if (not foundAnyMatchingExistingViews and not foundMatching3DPoint) {
            //This point did not match any existing cloud points - add it as new.
            mReconstructionCloud.push_back(p);
            newPoints++;
        }
    }

    if (mConsoleDebugLevel <= LOG_DEBUG) {
        cout << " adding: " << cloud.size() << " (new: " << newPoints << ", merged: " << mergedPoints << ")" << endl;
    }*/
}


void SFM::denseReconstruction(int method){
	/*
	for(int i = 0; i < matches.size(); i++){
		defImageSize = GRAY_imgs[i].size();
		Mat R1, R2, P1, P2, Q, RECT1, RECT2, map11, map12, map21, map22, K1RECT, K2RECT;
		stereoRectify(K1, Mat(distort1), K2, Mat(distort2), defImageSize, RtMatrices[i].first, RtMatrices[i].second, R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY, -1, defImageSize);
		
		initUndistortRectifyMap(K1, distort1, R1, K1RECT, GRAY_imgs[i].size(), CV_32FC1, map11, map12);
		initUndistortRectifyMap(K2, distort2, R2, K2RECT, GRAY_imgs[i+1].size(), CV_32FC1, map21, map22);
		
		remap(GRAY_imgs[i],   RECT1, map11, map12, INTER_CUBIC, BORDER_CONSTANT, 0);
		remap(GRAY_imgs[i+1], RECT2, map21, map22, INTER_CUBIC, BORDER_CONSTANT, 0);

		cv::imshow("Rect1", RECT1);
		cv::imshow("Rect2", RECT2);
		waitKey(-1);
	}*/
}

void SFM::visualizeCameraMotion(){
	viz::Viz3d window("Coordinate Frame");

	bool camera_pov = false;

    /// Add coordinate axes
    window.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    vector<Affine3d> path;

    Affine3d n = Affine3d(Matx33d(1,0,0,0,1,0,0,0,1),Mat(Matx31d(0,0,0)));

    path.push_back(Affine3d(n.rotation(), n.translation()));
	for (int i = 0; i < GRAY_imgs.size()-1; i++){
		for(int j = i+1; j < GRAY_imgs.size(); j++){
			map< intPair, MatPair >::iterator RtIt;
			if((RtIt = RtMap.find(intPair(i,j))) != RtMap.end()){
				n = Affine3d(Matx33d(1,0,0,0,1,0,0,0,1),RtIt->second.second)*n;
				n = Affine3d(RtIt->second.first, 0) * n;
				path.push_back(Affine3d(n.rotation(), n.translation()));
			}
		}
		
	}

    viz::WTrajectory trajectory(path, viz::WTrajectory::PATH, 0.5);
    viz::WTrajectoryFrustums frustums(path, Vec2f(0.889484, 0.523599), 0.5, viz::Color::yellow());


    window.showWidget("cameras", trajectory);
	window.showWidget("frustums", frustums);

    /// Start event loop.
    window.spin();
	
}

void SFM::showTriangulation(int i, int j){

}

void SFM::showEpipolarLines(){
	/*
	for(int i = 0; i < matches.size(); i++){
		vector<Point2f> points1;
		vector<Point2f> points2;
		vector<float> similarities;
		for(int k = 0; k < matches[i].size() && k < masks[i].total(); k++){
			if(masks[i].at<uchar>(k) != 0){
				points1.push_back(keypoints[i][matches[i][k].queryIdx].pt);
				points2.push_back(keypoints[i+1][matches[i][k].trainIdx].pt);
				similarities.push_back(matches[i][k].distance);
			}
		}

		Mat im1, im2;
		GRAY_imgs[i].copyTo(im1);
		GRAY_imgs[i+1].copyTo(im2);

		Size s1 = GRAY_imgs[i].size();
		Size s2 = GRAY_imgs[i+1].size();

		for(int h = 0; h < similarities.size(); h++){
			Mat l1 = fundMatrices[i].t() * Mat(Matx31d(points2[h].x, points2[h].y,1));
			Mat l2 = fundMatrices[i] * Mat(Matx31d(points1[h].x, points1[h].y,1));
			
			double l11, l12, l13, l21, l22, l23;

			if(l1.type() == CV_32FC1 && l2.type() == CV_32FC1){
				l11 = l1.at<float>(0);
				l12 = l1.at<float>(1);
				l13 = l1.at<float>(2);
				l21 = l2.at<float>(0);
				l22 = l2.at<float>(1);
				l23 = l2.at<float>(2);
			} else if(l1.type() == CV_64FC1 && l2.type() == CV_64FC1){
				l11 = l1.at<double>(0);
				l12 = l1.at<double>(1);
				l13 = l1.at<double>(2);
				l21 = l2.at<double>(0);
				l22 = l2.at<double>(1);
				l23 = l2.at<double>(2);
			}

			if(l12 < l11){
				line(im1, Point((-l13 -l12*0)/l11,0), Point( (-l13 -l12*s1.height)/l11, s1.height), Scalar(1,1,1), 1, 8, 0);
			} else {
				line(im1, Point(0, (-l13 -l11*0)/l12), Point(s1.width, (-l13 -l11*s1.width)/l12), Scalar(1,1,1), 1, 8, 0);
			}

			if(l22 < l21){
				line(im2, Point((-l23 -l22*0)/l21,0), Point( (-l23 -l22*s2.height)/l21, s2.height), Scalar(1,1,1), 1, 8, 0);
			} else {
				line(im2, Point(0, (-l23 -l21*0)/l22), Point(s2.width, (-l23 -l21*s2.width)/l22), Scalar(1,1,1), 1, 8, 0);
			}

			

		}
		cv::imshow( "Left", im1 );
		cv::imshow( "Right", im2 );
  		waitKey(-1);
	}*/
}

void SFM::showCorrespondences(){
	for(map< intPair, Mat >::iterator iter = maskMap.begin(); iter != maskMap.end(); iter++){

		intPair pr = iter->first;
		Mat localMask = iter->second;

		vector<DMatch> good_matches;
		map< intPair, vector<DMatch> >::iterator matchIt;
		if( (matchIt = matchMap.find(pr)) != matchMap.end()){
			vector<DMatch> localMatches = matchIt->second;

			for(int l = 0; l < localMatches.size(); l++){
				if(localMask.at<uchar>(l)){
					good_matches.push_back(localMatches[l]);
				}
			}
			
			cout << countNonZero(localMask) << ":" << localMask.total() << endl;

			Mat img_matches;
			drawMatches( GRAY_imgs[pr.first], keypoints[pr.first], GRAY_imgs[pr.second], keypoints[pr.second],
		               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

			Size size(img_matches.cols/2, img_matches.rows/2);
			Mat dst;
			resize(img_matches,dst,size);
		  	// Show detected matches

			cv::imshow( "Good Matches", dst );
	  		waitKey(-1);

	 	}
	}
}
