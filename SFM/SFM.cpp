#include <opencv2/viz/vizcore.hpp>
#include <opencv2/sfm.hpp>
#include <string>
#include <queue>
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
			mImageFeatures[i].points[j] = Point2d(kpts[j].pt);
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
	const float ratio = LOWE_MATCH_FILTER_RATIO;
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

	sortedMasks = vector< pair< intPair,Mat > >();
	for(map< intPair, Mat >::iterator miter = maskMap.begin(); miter != maskMap.end(); miter++){
		sortedMasks.push_back(*miter);
	}
	sort(sortedMasks.begin(), sortedMasks.end(), sortMasks);
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
				cv::sfm::essentialFromFundamental(F,K1,K2,E);
				essenMap.insert( pair< intPair,Mat >(pr,E) );
				vector<Mat> Rs;
				vector<Mat> ts;
				Mat bvp1=Mat(2,1,CV_64FC1), bvp2=Mat(2,1,CV_64FC1);
				bvp1.at<double>(0) = bestValidPoint1[0].x;
				bvp1.at<double>(1) = bestValidPoint1[0].y;
				bvp2.at<double>(0) = bestValidPoint2[0].x;
				bvp2.at<double>(1) = bestValidPoint2[0].y;
				cv::sfm::motionFromEssential(E, Rs, ts);
				int idx = cv::sfm::motionFromEssentialChooseSolution(Rs,ts,K1,bvp1,K2,bvp2);
				RtMap.insert( pair< intPair,MatPair >( pr,MatPair(Rs[idx],ts[idx]) ) );
			}
		}
	}
}

void SFM::initialBundleAdjustment(){
	for(int i = 0; i < sortedMasks.size(); i++){
		//If found Rotation and translation triangulate views

		intPair pr = sortedMasks[i].first;
		map< intPair, vector<DMatch> >::iterator mchIt;
		map< intPair, MatPair >::iterator RtIt;
		if( ((RtIt = RtMap.find(pr)) != RtMap.end()) && ((mchIt = matchMap.find(pr)) != matchMap.end()) ){
			Mat firstR = RtIt->second.first, t = RtIt->second.second;
			Matx34d Pleft = Matx34d::eye(), Pright;
    		Pright = Matx34d(firstR.at<double>(0,0), firstR.at<double>(0,1), firstR.at<double>(0,2), t.at<double>(0),
			                 firstR.at<double>(1,0), firstR.at<double>(1,1), firstR.at<double>(1,2), t.at<double>(1),
			                 firstR.at<double>(2,0), firstR.at<double>(2,1), firstR.at<double>(2,2), t.at<double>(2));
			
    		vector< DMatch > mch = mchIt->second;
    		vector<Point2d> points1;
			vector<Point2d> points2;
			vector<intPair> indxs;

			PointCloud_d pointCloud;

			for(int k = 0; k < mch.size(); k++){
				points1.push_back(Point2d(keypoints[pr.first][mch[k].queryIdx].pt));
				points2.push_back(Point2d(keypoints[pr.second][mch[k].trainIdx].pt));
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

/*
void SFM::additionalBundleAdjustment(){
	for(int i = 0; i < sortedMasks.size(); i++){
		//If found Rotation and translation triangulate views

		intPair pr = sortedMasks[i].first;
		map< intPair, vector<DMatch> >::iterator mchIt;
		map< intPair, MatPair >::iterator RtIt;
		if( ((RtIt = RtMap.find(pr)) != RtMap.end()) && ((mchIt = matchMap.find(pr)) != matchMap.end()) ){
			Mat firstR = RtIt->second.first, t = RtIt->second.second;
			Matx34d Pleft = Matx34d::eye(), Pright;
    		Pright = Matx34d(firstR.at<double>(0,0), firstR.at<double>(0,1), firstR.at<double>(0,2), t.at<double>(0),
			                 firstR.at<double>(1,0), firstR.at<double>(1,1), firstR.at<double>(1,2), t.at<double>(1),
			                 firstR.at<double>(2,0), firstR.at<double>(2,1), firstR.at<double>(2,2), t.at<double>(2));
			
    		vector< DMatch > mch = mchIt->second;
    		vector<Point2f> points1;
			vector<Point2f> points2;
			vector<intPair> indxs;

			PointCloud_d pointCloud;

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
*/
map<int,int> SFM::dependencyNet(){
	map<int,int> mMatches = map<int,int>();
	
	set<int>::iterator it;
	map<intPair, vector<DMatch> >::iterator mitr;
	for(it = mDoneViews.begin(); it != mDoneViews.end(); it++){
		int i = *it;
		for(int j = 0; j < GRAY_imgs.size(); j++){
			if((mitr = matchMap.find(intPair(i,j))) != matchMap.end()){

				//mMatches.push_back();
			}
			if((mitr = matchMap.find(intPair(j,i))) != matchMap.end()){
				//mMatches.push_back();
			}
		}
	}

	return mMatches;
}

void SFM::incrementalBundleAdjustment() {
	/*
    while (mDoneViews.size() != GRAY_imgs.size()) {
        //Find the best view to add, according to the largest number of 2D-3D corresponding points
        intPair matches2D3D = maxNumMatches();

        size_t bestView;
        size_t bestNumMatches = 0;
        for (const auto& match2D3D : matches2D3D) {
			const size_t numMatches = match2D3D.second.points2D.size();
			if (numMatches > bestNumMatches) {
                bestView       = match2D3D.first;
                bestNumMatches = numMatches;
            }
        }

        #if defined VERBOSE
        cout << "Best view " << bestView << " has " << bestNumMatches << " matches" << endl;
        cout << "Adding " << bestView << " to existing " << Mat(vector<int>(mGoodViews.begin(), mGoodViews.end())).t() << endl;
        #endif
        mDoneViews.insert(bestView);

        //recover the new view camera pose
        Matx34f newCameraPose;
        bool success = SfMStereoUtilities::findCameraPoseFrom2D3DMatch(
                mIntrinsics,
                matches2D3D[bestView],
                newCameraPose);

        if (not success) {
            if (mConsoleDebugLevel <= LOG_WARN) {
                cerr << "Cannot recover camera pose for view " << bestView << endl;
            }
            continue;
        }

        mCameraPoses[bestView] = newCameraPose;

        if (mConsoleDebugLevel <= LOG_DEBUG) {
            cout << "New view " << bestView << " pose " << endl << newCameraPose << endl;
        }

        //triangulate more points from new view to all existing good views
        bool anyViewSuccess = false;
        for (const int goodView : mGoodViews) {
            //since match matrix is upper-triangular (non symmetric) - use lower index as left
            size_t leftViewIdx  = (goodView < bestView) ? goodView : bestView;
            size_t rightViewIdx = (goodView < bestView) ? bestView : goodView;

            Matching prunedMatching;
            Matx34f Pleft  = Matx34f::eye();
            Matx34f Pright = Matx34f::eye();

            //use the essential matrix recovery to prune the matches
            bool success = SfMStereoUtilities::findCameraMatricesFromMatch(
                    mIntrinsics,
                    mFeatureMatchMatrix[leftViewIdx][rightViewIdx],
                    mImageFeatures[leftViewIdx],
                    mImageFeatures[rightViewIdx],
    				prunedMatching,
                    Pleft, Pright
                    );
            mFeatureMatchMatrix[leftViewIdx][rightViewIdx] = prunedMatching;

            //triangulate the matching points
            PointCloud pointCloud;
            success = SfMStereoUtilities::triangulateViews(
                    mIntrinsics,
                    { leftViewIdx, rightViewIdx },
                    mFeatureMatchMatrix[leftViewIdx][rightViewIdx],
                    mImageFeatures[leftViewIdx],
                    mImageFeatures[rightViewIdx],
                    mCameraPoses[leftViewIdx],
                    mCameraPoses[rightViewIdx],
                    pointCloud
                    );

            if (success) {
                if (mConsoleDebugLevel <= LOG_DEBUG) {
                    cout << "Merge triangulation between " << leftViewIdx << " and " << rightViewIdx <<
                        " (# matching pts = " << (mFeatureMatchMatrix[leftViewIdx][rightViewIdx].size()) << ") ";
                }

                //add new points to the reconstruction
                mergeNewPointCloud(pointCloud);

                anyViewSuccess = true;
            } else {
                if (mConsoleDebugLevel <= LOG_WARN) {
                    cerr << "Failed to triangulate " << leftViewIdx << " and " << rightViewIdx << endl;
                }
            }
        }

        if (anyViewSuccess) {
            ba.run(mReconstructionCloud, mCameraPoses, K1, mImageFeatures, BA_RE_SIMPLE);
        }
        mGoodViews.insert(bestView);
    }*/
}

void SFM::mergePointCloud(const PointCloud_d& cloud) {
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

	
    viz::Viz3d window2("Coordinate Frame Translation-Rotation");


    /// Add coordinate axes
    window2.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    vector<Affine3d> path2;

    Affine3d n2 = Affine3d(Matx33d::eye(),Mat(Matx31d(0,0,0)));


    path2.push_back(Affine3d(n2.rotation(), n2.translation()));
	for (int i = 0; i < GRAY_imgs.size()-1; i++){
		for(int j = i+1; j < GRAY_imgs.size(); j++){
			map< intPair, MatPair >::iterator RtIt;
			if((RtIt = RtMap.find(intPair(i,j))) != RtMap.end()){
				n2 = Affine3d(Mat(Matx33d::eye()),RtIt->second.second) *n2;
				n2 = Affine3d(RtIt->second.first,Mat(Matx31d(0,0,0)))*n2;
				
				path2.push_back(Affine3d(n2.rotation(), n2.translation()));
				#if defined VERBOSE
				cout << "Added view motion " << i << "|" << j << " to visualization" << endl;
				#endif
			}
		}
		
	}

    viz::WTrajectory trajectory2(path2, viz::WTrajectory::PATH, 0.5);
    viz::WTrajectoryFrustums frustums2(path2, Vec2f(0.889484, 0.523599), 0.5, viz::Color::yellow());


    vector<Point3f> localCloud = vector<Point3f>();
    cout << "cloudSize: " << mReconstructionCloud.size() << endl;
    for(int i = 0; i < mReconstructionCloud.size(); i++){
    	localCloud.push_back(mReconstructionCloud[i].p);
    }
    window2.showWidget("cloud", WCloud(localCloud));
    window2.showWidget("cameras", trajectory2);
	window2.showWidget("frustums", frustums2);

	Point3d cam_pos(8.0,8.0,8.0), cam_focal_point(-1.0,-1.0,-1.0), cam_y_dir(-1.0,0.0,0.0);
    Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
	window2.setViewerPose(cam_pose);

    /// Start event loop.
    window2.spin();

}

void SFM::showTriangulation(int first, int last, float max_dist){
	CV_Assert(first < last);

	viz::Viz3d window("Point Cloud");
    /// Add coordinate axes
    window.showWidget("Coordinate Widget", viz::WCoordinateSystem());

	Affine3d n = Affine3d(Matx33d::eye(), Mat(Matx31d(0,0,0)));
	vector<Affine3d> path;
	RNG rng(12345);

	vector<Mat> outp = vector<Mat>();
	vector< Mat > mr = vector< Mat >();
	vector< Mat > mt = vector< Mat >();

	path.push_back(Affine3d(n.rotation(),n.translation()));
	mr.push_back(Mat(n.rotation()));
	mt.push_back(Mat(n.translation()));
	for(int i = first; i < last; i++){
		intPair p = intPair(i,i+1);
		map< intPair, MatPair >::iterator RtIt;
		if((RtIt = RtMap.find(p)) != RtMap.end()){
			n = Affine3d(RtIt->second.first, RtIt->second.second)*n;
			cout << n.rotation() << n.translation() << endl;
			mr.push_back(Mat(n.rotation()));
			mt.push_back(Mat(n.translation()));
			path.push_back(Affine3d(n.rotation(),n.translation()));
			#if defined VERBOSE
			cout << "Added view motion " << i << "|" << i+1 << " to visualization" << endl;
			#endif
		}
	}

	viz::WTrajectory trajectory(path, viz::WTrajectory::PATH, 0.5);
    viz::WTrajectoryFrustums frustums(path, Vec2f(0.889484, 0.523599), 0.5, viz::Color::yellow());

    window.showWidget("cameras", trajectory);
	window.showWidget("frustums", frustums);

	for(int i = 0; i < path.size()-1; i++){
		map< intPair, MatPair >::iterator RtIt2;

		vector<Mat> Ps = vector<Mat>(2);
		hconcat(mr[i], mt[i], Ps[0]);
		hconcat(mr[i+1], mt[i+1], Ps[1]);
		//cout << Ps[0] << endl;
		//cout << Ps[1] << endl;

		vector< vector<Point2d> > pts = vector< vector<Point2d> >(2);

		intPair pr = intPair(i,i+1);
		map< intPair, Mat >::iterator fundIt;
		map< intPair, vector<DMatch> >::iterator matchIt;
		if((fundIt = maskMap.find(pr)) != maskMap.end() && (matchIt = matchMap.find(pr)) != matchMap.end()){
			vector<DMatch> mch = matchIt->second;
			for(int k = 0; k < mch.size(); k++){
				pts[0].push_back(Point2d(keypoints[pr.first][mch[k].queryIdx].pt));
				pts[1].push_back(Point2d(keypoints[pr.second][mch[k].trainIdx].pt));
			}
			
			Mat points3f;
			cout << "triangulating" << endl;
			if(Estimator::triangulateViews(pts[0], pts[1], Ps[0], Ps[1], K1, points3f)){


				cout << "adding new to cloud" << endl;
				outp.push_back(points3f);
			}
		}
	}

	for(int hh = 0; hh < outp.size(); hh++){
		stringstream ss;
		ss << hh;
		window.showWidget(string("all") + ss.str(), WCloud(outp[hh].t()));
	}
	/// Start event loop.
    window.spin();

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
