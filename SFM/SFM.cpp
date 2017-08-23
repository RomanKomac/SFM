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
	mCameraPosesValid.resize(GRAY_imgs.size());
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
	mCameraPosesValid.resize(GRAY_imgs.size());
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
	mCameraPosesValid.resize(GRAY_imgs.size());
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
	mCameraPosesValid.resize(GRAY_imgs.size());
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
	mCameraPosesValid.resize(GRAY_imgs.size());
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
	mCameraPosesValid.resize(GRAY_imgs.size());
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

			//Front reference, Back reference
			map< int,int > fRef, bRef;
			for(int v = 0; v < mask.total(); v++){
				if(mask.at<uchar>(v) != 0){
					int idxI = mch[v].queryIdx;
					int idxJ = mch[v].trainIdx;
					fRef.insert(intPair(idxI,idxJ));
					bRef.insert(intPair(idxJ,idxI));
				}
			}
			connectionsMap[intPair(pr.first,pr.second)] = fRef;
			connectionsMap[intPair(pr.second,pr.first)] = bRef;
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

				Mat E, Et, R, t, F = fundIt->second;
				cv::sfm::essentialFromFundamental(F,K1,K2,E);
				cv::sfm::essentialFromFundamental(F.t(),K1,K2,Et);
				essenMap.insert( pair< intPair,Mat >(pr,E) );
				essenMap.insert( pair< intPair,Mat >(intPair(pr.second,pr.first),Et) );
				vector<Mat> Rs, Rst;
				vector<Mat> ts, tst;
				Mat bvp1=Mat(2,1,CV_64FC1), bvp2=Mat(2,1,CV_64FC1);
				bvp1.at<double>(0) = bestValidPoint1[0].x;
				bvp1.at<double>(1) = bestValidPoint1[0].y;
				bvp2.at<double>(0) = bestValidPoint2[0].x;
				bvp2.at<double>(1) = bestValidPoint2[0].y;
				cv::sfm::motionFromEssential(E, Rs, ts);
				cv::sfm::motionFromEssential(Et, Rst, tst);
				int idx = cv::sfm::motionFromEssentialChooseSolution(Rs,ts,K1,bvp1,K2,bvp2);
				int idxt = cv::sfm::motionFromEssentialChooseSolution(Rst,tst,K2,bvp2,K1,bvp1);
				RtMap.insert( pair< intPair,MatPair >( pr,MatPair(Rs[idx],ts[idx]) ) );
				RtMap.insert( pair< intPair,MatPair >( intPair(pr.second,pr.first),MatPair(Rst[idxt],tst[idxt]) ) );
			}
		}
	}
}

void SFM::additionalBundleAdjustment(){
	int hch = 0;
	while (mDoneViews.size() != GRAY_imgs.size() && hch < 6) {
		hch++;
        //Find the best views to add
        map< int,intPair > matches2D3D = dependencyNet();

        //recover the new view camera pose
        Matx34d Pleft, Pright;
        map< int,intPair >::const_iterator miter;
        for(miter = matches2D3D.begin(); miter != matches2D3D.end(); miter++){
        	intPair ip = miter->second;
        	int i = miter->first;

        	cout << ip.first << "|" << ip.second << endl;
        	vector<double> coeffs;
        	vector<Point3d> p3d;
        	vector<Point2d> p2d;
        	if(i == ip.first){
        		MatPair Rt = RtMap[ip];
        		Affine3d a3d = Affine3d(Rt.first, Rt.second) * Affine3d( Mat(mCameraPoses[i].get_minor<3, 3>(0, 0)), Mat(mCameraPoses[i].get_minor<3, 1>(0, 3)) );
        		Pleft  = mCameraPoses[i];

        		hconcat(a3d.rotation(), a3d.translation(), Pright);
        		retrieveCorrespondences(p3d, p2d, ip.first, ip.second);

        		Mat rvec, rMat, tvec;
        		if(p3d.size() < 4)
        			continue;
        		bool PnPsuccess = solvePnP(p3d, p2d, K1, coeffs, rvec, tvec, false, CV_EPNP);
        		Rodrigues(rvec, rMat);
        		cout << "PnP " << (PnPsuccess? "successful":"unsuccessful") << endl;
        		hconcat(rMat, tvec, Pright);
        	} else {

        		MatPair Rt = RtMap[ip];
        		Affine3d a3d = Affine3d(Rt.first, Rt.second).inv() * Affine3d( Mat(mCameraPoses[i].get_minor<3, 3>(0, 0)), Mat(mCameraPoses[i].get_minor<3, 1>(0, 3)) );
        		Pright = mCameraPoses[i];

        		hconcat(a3d.rotation(), a3d.translation(), Pleft);
        		retrieveCorrespondences(p3d, p2d, ip.second, ip.first);

        		Mat rvec, rMat, tvec;
        		if(p3d.size() < 4)
        			continue;
        		bool PnPsuccess = solvePnP(p3d, p2d, K1, coeffs, rvec, tvec, false, CV_EPNP);
				Rodrigues(rvec, rMat);
        		cout << "PnP " << (PnPsuccess? "successful":"unsuccessful") << endl;
        		hconcat(rMat, tvec, Pleft);
        	}

        	vector< DMatch > mch = matchMap[ip];
    		vector<Point2d> points1;
			vector<Point2d> points2;
			vector<intPair> indxs;
			PointCloud_d pointCloud;

			for(int k = 0; k < mch.size(); k++){
				points1.push_back(mImageFeatures[ip.first].points[mch[k].queryIdx]);
				points2.push_back(mImageFeatures[ip.second].points[mch[k].queryIdx]);
				indxs.push_back(intPair(mch[k].queryIdx, mch[k].trainIdx));
			}

			if(Estimator::triangulateViews(points1, points2, Pleft, Pright, indxs, ip, K1, pointCloud)){
				mReconstructionCloud.insert( mReconstructionCloud.end(), pointCloud.begin(), pointCloud.end() );
				mergePointCloud();
				mCameraPoses[ip.first]  = Pleft;
				mCameraPoses[ip.second] = Pright;
				mDoneViews.insert(ip.first);
				mDoneViews.insert(ip.second);
				mGoodViews.insert(ip.first);
				mGoodViews.insert(ip.second);
				
			} else {
				//If points could not be triangulated from views
				continue;
			}
        }
        ba.run(mReconstructionCloud, mCameraPoses, K1, mImageFeatures, BA_RE_SIMPLE);
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
			hconcat(firstR, t, Pright);
			
    		vector< DMatch > mch = mchIt->second;
    		vector<Point2d> points1;
			vector<Point2d> points2;
			vector<intPair> indxs;

			PointCloud_d pointCloud;
			cout << pr.first << "|" << pr.second << endl;

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


map< int,intPair > SFM::dependencyNet(){
	map< int,intPair > mMatches = map< int,intPair >();
	
	set<int>::iterator it;
	map<intPair, vector<DMatch> >::iterator mitr;
	for(it = mDoneViews.begin(); it != mDoneViews.end(); it++){
		int i = *it;
		for(int j = 0; j < GRAY_imgs.size(); j++){
			if((mitr = matchMap.find(intPair(i,j))) != matchMap.end() && (mDoneViews.find(j) == mDoneViews.end()) ){
				mMatches[i] = intPair(i,j);
			}
			if((mitr = matchMap.find(intPair(j,i))) != matchMap.end() && (mDoneViews.find(j) == mDoneViews.end())){
				mMatches[i] = intPair(j,i);
			}
		}
	}

	return mMatches;
}


void SFM::mergePointCloud() {
    int mergedPoints = 0;
    int cpoints = 0;
    PointCloud_d holder;
    map<int, int>::const_iterator mit, mit2;

    //Mask for used points
    vector<bool> cloudMask = vector<bool>(mReconstructionCloud.size());

    map< intPair,vector<int> > pointTo3D;
    map< intPair,set< intPair > > connected;
    for(int i = 0; i < mReconstructionCloud.size(); i++){
    	cloudMask[i] = true;
    	Mapped3DPoint_d pcit = mReconstructionCloud[i];
    	for(mit = (pcit.views).begin(); mit != (pcit.views).end(); mit++){
    		pointTo3D[(*mit)].push_back(i);
    		for(mit2 = (pcit.views).begin(); mit2 != (pcit.views).end(); mit2++){
    			connected[(*mit)].insert((*mit2));
    		}
    	}	
    }

    
    //Dense net of views
    map< intPair,set< intPair > >::iterator conIter;
    for(conIter = connected.begin(); conIter != connected.end(); conIter++){
    	deque< intPair > dq;
    	dq.push_back((conIter->first));
    	set< intPair > sip;
    	set< intPair >::const_iterator sit;
    	while(!dq.empty()){
    		intPair b = dq.front();
    		if(sip.count(b) == 0){
    			sip.insert(b);
    			for(sit = connected[b].begin(); sit != connected[b].end(); sit++){
    				dq.push_back((*sit));
    			}
    		}
    		dq.pop_front();
    	}
    	conIter->second = sip;
    	//cout << (conIter->first).first << "|" << (conIter->first).second << "|" << sip.size() << endl;
    }

    //Retrieving points from across all views
    
    for(int j = 0; j < mReconstructionCloud.size(); j++){
    	//If current point hasn't been assigned yet
    	if(cloudMask[j]){
    		PointCloud_d refs;

    		map< int,int >::const_iterator v,vinner;
			v = mReconstructionCloud[j].views.begin();
			set< intPair > ip = connected[(*v)];
    		for(int k = 0; k < mReconstructionCloud.size(); k++){
    			vinner = mReconstructionCloud[k].views.begin();
    			if(ip.count((*vinner)) != 0){
    				refs.push_back( mReconstructionCloud[k] );
    				//Point is already merged
    				cloudMask[k] = false;

    			}
    		}
    		mergedPoints++;

    		//Retrieved references merge
    		//vector<*Mapped3DPoint_d>::const_iterator citv;
    		Point3d p3d;
    		for(int l = 0; l < refs.size(); l++){
    			p3d+=refs[l].p;
    		}
    		//Average
    		p3d /= (int)refs.size();
    		Mapped3DPoint_d m3dp;
    		m3dp.p = p3d;

    		set< intPair >::const_iterator csit;
    		for(csit = ip.begin(); csit != ip.end(); csit++)
    			m3dp.views.insert(*csit);

    		
    		if(m3dp.views.size() >= 2){
    			if(cpoints > 1000)
    				continue;
    			cpoints++;
    		}
    		holder.push_back(m3dp);
    		
    	}
    }
    
    mReconstructionCloud.swap(holder);

    #if defined VERBOSE
    cout << "# points after merging: " << mergedPoints << endl;
    #endif
}

void SFM::visualizeCameraMotion(){
    viz::Viz3d window2("Coordinate Frame Translation-Rotation");

    /// Add coordinate axes
    //window2.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    vector<Affine3d> path2;

    for(int i = 0; i < mCameraPoses.size(); i++){
    	path2.push_back( Affine3d( Mat(mCameraPoses[i].get_minor<3, 3>(0, 0)), Mat(mCameraPoses[i].get_minor<3, 1>(0, 3)))  );
    }

    Point3d pmiddle;
    for(int j = 0; j < mReconstructionCloud.size(); j++){
    	pmiddle += mReconstructionCloud[j].p;
    }
    pmiddle/=(int)mReconstructionCloud.size();

    viz::WTrajectory trajectory2(path2, viz::WTrajectory::PATH, 5);
    viz::WTrajectoryFrustums frustums2(path2, Vec2f(0.445, 0.267), 0.05, viz::Color::yellow());

    if(mReconstructionCloud.size() != 0){
	    vector<Point3f> localCloud = vector<Point3f>();
	    cout << "cloudSize: " << mReconstructionCloud.size() << endl;
	    
	    for(int i = 0; i < mReconstructionCloud.size(); i++){
	    	localCloud.push_back(mReconstructionCloud[i].p);
	    }
	    window2.showWidget("cloud", WCloud(localCloud));
	}
    window2.showWidget("cameras", trajectory2);
	window2.showWidget("frustums", frustums2);

	Point3d cam_pos(pmiddle.x,pmiddle.y,pmiddle.z), cam_focal_point(-1.0,-1.0,-1.0), cam_y_dir(-1.0,0.0,0.0);
    Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
	window2.setViewerPose(cam_pose);

    /// Start event loop.
    window2.spin();

}

void SFM::triangulationMinThreeViews(){
	//Set all camera poses to invalid, to regenerate from scratch
	for(int l = 0; l < mCameraPosesValid.size(); l++)
		mCameraPosesValid[l] = false;

	mCameraPosesValid[0] = true;
	mCameraPoses[0] = Matx34d(1,0,0,0,0,1,0,0,0,0,1,0);
	for(int i = 1; i < mCameraPoses.size()-1; i++){
		int central_view = i;
		set<int> c_points = consistencies[i];

		//Points of views. (0-1)(previous-current) (1-2)(current-next)
		vector< vector<Point2d> > pts = vector< vector<Point2d> >(3);

		//Indices of points
		vector< vector< intPair > > indxs = vector< vector< intPair > >(2);

		//Pair of indices of views
		vector< intPair > pr = vector< intPair >(2);
		pr[0] = intPair(central_view-1, central_view);
		pr[1] = intPair(central_view, central_view+1);

		//Retrieving indices of view pairs
		map< intPair,map< int,int > >::iterator cit1, cit2, h;
		bool fb = false, sb = false;

		cout << "Obtaining views" << endl;
		for(h = connectionsMap.begin(); h != connectionsMap.end(); h++){
			if( (h->first) == intPair(central_view,central_view-1) ){cit1 = h; fb = true;}
			if( (h->first) == intPair(central_view,central_view+1) ){cit2 = h; sb = true;}
		}

		if(!fb || !sb)
			continue;

		map<int,int> m1 = cit1->second;
		map<int,int> m2 = cit2->second;

		//Iterating unique indices set
		set<int>::iterator sit;
		int ctr = 0;
		for(sit = c_points.begin(); sit != c_points.end(); sit++){
			int i2 = (*sit);
			int i1 = m1[i2];
			int i3 = m2[i2];
			pts[0].push_back(Point2d(keypoints[central_view-1][i1].pt));
			pts[1].push_back(Point2d(keypoints[central_view  ][i2].pt));
			pts[2].push_back(Point2d(keypoints[central_view+1][i3].pt));
			indxs[0].push_back(intPair(i1, i2));
			indxs[1].push_back(intPair(i2, i3));
			ctr++;
		}

		if(!mCameraPosesValid[central_view]){
			mCameraPosesValid[central_view] = true;
			Matx34d cbef = mCameraPoses[central_view-1];
			Affine3d aftr, abef = Affine3d( Mat(cbef.get_minor<3, 3>(0, 0)), Mat(cbef.get_minor<3, 1>(0, 3)) );
			MatPair mp = RtMap[intPair(central_view-1,central_view)];
			aftr = Affine3d( mp.first, mp.second ) * abef;
			hconcat(aftr.rotation(), aftr.translation(), mCameraPoses[central_view]);
		}
		if(!mCameraPosesValid[central_view+1]){
			mCameraPosesValid[central_view+1] = true;
			Matx34d cbef = mCameraPoses[central_view];
			Affine3d aftr, abef = Affine3d( Mat(cbef.get_minor<3, 3>(0, 0)), Mat(cbef.get_minor<3, 1>(0, 3)) );
			MatPair mp = RtMap[intPair(central_view,central_view+1)];
			aftr = Affine3d( mp.first, mp.second ) * abef;
			hconcat(aftr.rotation(), aftr.translation(), mCameraPoses[central_view+1]);
		}

		//mReconstructionCloud.clear();

		PointCloud_d pointCloud1, pointCloud2;
		if(Estimator::triangulateViews(pts[0], pts[1], mCameraPoses[central_view-1], mCameraPoses[central_view],   indxs[0], pr[0], K1, pointCloud1)){
			for(int h = 0; h < pointCloud1.size(); h++){
				mReconstructionCloud.push_back(pointCloud1[h]);
			}
			//mReconstructionCloud.insert( mReconstructionCloud.end(), pointCloud1.begin(), pointCloud1.end() );
			#if defined VERBOSE
			cout << "inserted: " << pointCloud1.size() << " points for views: " << central_view-1 << "|" << central_view << endl;
			#endif
		}
		if(Estimator::triangulateViews(pts[1], pts[2], mCameraPoses[central_view],   mCameraPoses[central_view+1], indxs[1], pr[1], K1, pointCloud2)){
			for(int h = 0; h < pointCloud2.size(); h++){
				mReconstructionCloud.push_back(pointCloud2[h]);
			}
			//mReconstructionCloud.insert( mReconstructionCloud.end(), pointCloud2.begin(), pointCloud2.end() );
			#if defined VERBOSE
			cout << "inserted: " << pointCloud2.size() << " points for views: " << central_view << "|" << central_view+1 << endl;
			#endif
		}
		#if defined VERBOSE
		cout << "Current cloud size: " << mReconstructionCloud.size() << endl;
		#endif

		mergePointCloud();
		PointCloud_d holderPcl = PointCloud_d(mReconstructionCloud.size());
		copy(mReconstructionCloud.begin(), mReconstructionCloud.end(), holderPcl.begin());
		if(!ba.run(mReconstructionCloud, mCameraPoses, K1, mImageFeatures, BA_RE_SIMPLE)){
			mReconstructionCloud = holderPcl;
			break;
		}

	}

	/*
	map< int,set<int> >::iterator mit;
	for(mit = consistencies.begin(); mit != consistencies.end(); mit++){
		//Middle view
		int central_view = mit->first;
		//Indices of consistent points from central view
		set<int> c_points = mit->second;

		//Points of views. (0-1)(previous-current) (1-2)(current-next)
		vector< vector<Point2d> > pts = vector< vector<Point2d> >(3);

		//Indices of points
		vector< vector< intPair > > indxs = vector< vector< intPair > >(2);

		//Pair of indices of views
		vector< intPair > pr = vector< intPair >(2);
		pr[0] = intPair(central_view-1, central_view);
		pr[1] = intPair(central_view, central_view+1);

		//Retrieving indices of view pairs
		map< intPair,map< int,int > >::iterator cit1, cit2, h;
		bool fb = false, sb = false;

		cout << "Obtaining views" << endl;
		for(h = connectionsMap.begin(); h != connectionsMap.end(); h++){
			if( (h->first) == intPair(central_view,central_view-1) ){cit1 = h; fb = true;}
			if( (h->first) == intPair(central_view,central_view+1) ){cit2 = h; sb = true;}
		}

		if(!fb || !sb)
			continue;

		map<int,int> m1 = cit1->second;
		map<int,int> m2 = cit2->second;

		//Iterating unique indices set
		set<int>::iterator sit;
		int ctr = 0;
		for(sit = c_points.begin(); sit != c_points.end(); sit++){
			int i2 = (*sit);
			int i1 = m1[i2];
			int i3 = m2[i2];
			pts[0].push_back(Point2d(keypoints[central_view-1][i1].pt));
			pts[1].push_back(Point2d(keypoints[central_view  ][i2].pt));
			pts[2].push_back(Point2d(keypoints[central_view+1][i3].pt));
			indxs[0].push_back(intPair(i1, i2));
			indxs[1].push_back(intPair(i2, i3));
			ctr++;
		}

		PointCloud_d pointCloud1, pointCloud2;
		if(Estimator::triangulateViews(pts[0], pts[1], mCameraPoses[central_view-1], mCameraPoses[central_view],   indxs[0], pr[0], K1, pointCloud1)){
			mReconstructionCloud.insert( mReconstructionCloud.end(), pointCloud1.begin(), pointCloud1.end() );
			#if defined VERBOSE
			cout << "inserted: " << pointCloud1.size() << " points for views: " << central_view-1 << "|" << central_view << endl;
			#endif
		}
		if(Estimator::triangulateViews(pts[1], pts[2], mCameraPoses[central_view],   mCameraPoses[central_view+1], indxs[1], pr[1], K1, pointCloud2)){
			mReconstructionCloud.insert( mReconstructionCloud.end(), pointCloud2.begin(), pointCloud2.end() );
			#if defined VERBOSE
			cout << "inserted: " << pointCloud2.size() << " points for views: " << central_view << "|" << central_view+1 << endl;
			#endif
		}
		#if defined VERBOSE
		cout << "Current cloud size: " << mReconstructionCloud.size() << endl;
		#endif
		mergePointCloud();
		ba.run(mReconstructionCloud, mCameraPoses, K1, mImageFeatures, BA_RE_SIMPLE);
	}*/
}

void SFM::sparseReconstruction(){

	for(int i = 0; i < mCameraPoses.size()-1; i++){
		PointCloud_d pcld;
		map< intPair,map< int,int > >::iterator cit;
		map< int,int >::const_iterator citer;
		map< int,int > mp = connectionsMap[intPair(i,i+1)];

		vector<Point2d> pts1, pts2;
		vector< intPair > indxs = vector< intPair >();

		for(citer = mp.begin(); citer != mp.end(); citer++){
			pts1.push_back(Point2d(keypoints[i][(citer->first)].pt));
			pts2.push_back(Point2d(keypoints[i+1][(citer->second)].pt));
			indxs.push_back(*citer);
		}

		if(Estimator::triangulateViews(pts1, pts2, mCameraPoses[i], mCameraPoses[i+1], indxs, intPair(i,i+1), K1, pcld)){
			cout << "Sparse recon. views: " << i << "|" << i+1 << endl;
			mReconstructionCloud.insert( mReconstructionCloud.end(), pcld.begin(), pcld.end() );
		}

	}
	cout << "cSize: " << mReconstructionCloud.size() << endl;
	//mergePointCloud();
}

void SFM::buildNet(){
	set< int > added;
	map< int,vector<int> > addedPairs;
	deque< int > dq, dq2;
	dq.push_back(0);
	while(!dq.empty()){
		int n = dq.front();
		dq.pop_front();
		if(added.count(n) == 0){
			for(int i = 0; i < sortedMasks.size(); i++){
				intPair ip = sortedMasks[i].first;
				if(ip.first == n && added.count(ip.second) == 0){
					dq.push_back(ip.second);
					addedPairs[ip.first].push_back(ip.second);
				} else if(ip.second == n && added.count(ip.first) == 0){
					dq.push_back(ip.first);
					addedPairs[ip.second].push_back(ip.first);
				}
			}
			added.insert(n);
			cout << n << endl;
		}
		
		
	}

	cout << "Affines" << endl;

	dq2.push_back(0);
	map< int,Affine3d > a3d;
	a3d.insert( pair< int,Affine3d>(0, Affine3d(Mat(Matx33d(1,0,0,0,1,0,0,0,1)), Mat(Matx31d(0,0,0))) ) );
	while(!dq2.empty()){
		int n = dq2.front();
		dq2.pop_front();

		for(int j = 0; j < addedPairs[n].size(); j++){
			int m = addedPairs[n][j];
			
			dq2.push_back(m);
			MatPair mp = RtMap[intPair(n,m)];
			Affine3d curraff = Affine3d(mp.first,mp.second) * a3d[n];
			//cout << curraff.rotation() << curraff.translation() << endl;
			a3d.insert(pair< int,Affine3d >(m,curraff));
		}
	}

	for(int i = 0; i < mCameraPoses.size(); i++){
		cout << a3d[i].rotation() << a3d[i].translation() << endl;
		hconcat(a3d[i].rotation(), a3d[i].translation(), mCameraPoses[i]);
		//cout << a3d[i].rotation() << a3d[i].translation() << endl;
	}

}

void SFM::minThreeViewsConsistency(){

	mReconstructionCloud.clear();
	

	for(int i = 1; i < mCameraPoses.size()-1; i++){
		vector< vector<Point2d> > pts = vector< vector<Point2d> >(2);
		vector< intPair > indxs = vector< intPair >();
		intPair pr = intPair(i,i+1);

		set<int> s1, s2, isect;
		map< intPair,map< int,int > >::iterator cit1, cit2;
		//connectionsMap contains only consistent points across views
		if( (cit1 = connectionsMap.find(intPair(i,i-1))) == connectionsMap.end() || 
			(cit2 = connectionsMap.find(intPair(i,i+1))) == connectionsMap.end())
			continue;
		cout << "Adding consistent relation of three views: " << i-1 << "|" << i << "|" << i+1 << endl;
		
		//Getting consistent indexes
		map< int,int >::iterator mit;
		map< int,int > m1 = cit1->second;
		map< int,int > m2 = cit2->second;
		for(mit = m1.begin(); mit != m1.end(); mit++){s1.insert(mit->first);}
		for(mit = m2.begin(); mit != m2.end(); mit++){s2.insert(mit->first);}
		set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), inserter(isect,isect.begin()));
		cout << "# of consistent points: " << isect.size() << endl;
		consistencies[i] = isect;
	}
}

void SFM::bundleAdjustment(){
	ba.run(mReconstructionCloud, mCameraPoses, K1, mImageFeatures, BA_RE_SIMPLE);
	/*
	mReconstructionCloud.clear();

	viz::Viz3d window("Point Cloud");
    window.showWidget("Coordinate Widget", viz::WCoordinateSystem());

	Affine3d n = Affine3d(Matx33d::eye(), Mat(Matx31d(0,0,0)));
	vector<Affine3d> path;

	//
	vector< Mat > outp = vector< Mat >();
	vector< Mat > mr = vector< Mat >();
	vector< Mat > mt = vector< Mat >();

	//Translates and rotates correctly
	path.push_back(Affine3d(n.rotation(),n.translation()));
	mr.push_back(Mat(n.rotation()));
	mt.push_back(Mat(n.translation()));
	for(int i = 0; i < GRAY_imgs.size()-1; i++){
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

	for(int i = 0; i < path.size(); i++){
		Mat pose;
		hconcat(mr[i], mt[i], pose);
		mCameraPoses[i] = pose;
	}

	for(int i = 0; i < path.size()-1; i++){
		map< intPair, MatPair >::iterator RtIt2;

		vector<Mat> Ps = vector<Mat>(2);
		Ps[0] = mCameraPoses[i];
		Ps[1] = mCameraPoses[i+1];

		vector< vector<Point2d> > pts = vector< vector<Point2d> >(2);
		vector< intPair > indxs = vector< intPair >();

		intPair pr = intPair(i,i+1);
		map< intPair, Mat >::iterator fundIt;
		map< intPair, vector<DMatch> >::iterator matchIt;
		if((fundIt = maskMap.find(pr)) != maskMap.end() && (matchIt = matchMap.find(pr)) != matchMap.end()){
			vector<DMatch> mch = matchIt->second;
			for(int k = 0; k < mch.size(); k++){
				pts[0].push_back(Point2d(keypoints[pr.first][mch[k].queryIdx].pt));
				pts[1].push_back(Point2d(keypoints[pr.second][mch[k].trainIdx].pt));
				indxs.push_back(intPair(mch[k].queryIdx, mch[k].trainIdx));
			}
			
			Mat points3f;
			cout << "triangulating" << endl;
			if(Estimator::triangulateViews(pts[0], pts[1], Ps[0], Ps[1], indxs, pr, K1, pointCloud)){
				mergePointCloud(pointCloud, 6);
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
	*/
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

void SFM::retrieveCorrespondences(vector<Point3d>& p3d, vector<Point2d>& p2d, int ref, int sel){
	p3d.clear();
	p2d.clear();
	PointCloud_d::const_iterator pclit;
	for(pclit = mReconstructionCloud.begin(); pclit != mReconstructionCloud.end(); pclit++){
		if((pclit->views).count(ref) != 0){
			map< int,int > mp = pclit->views;
			int idx = mp[ref];
			map< int,int > m = connectionsMap[intPair(ref,sel)];
			p2d.push_back(mImageFeatures[sel].points[(m[idx])]);
			p3d.push_back(pclit->p);
		}
	}
	cout << "connected 3d points: " << p3d.size() << endl;
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
