#include <stdio.h>
#include <string>
#include <vector>
#include <dirent.h>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include "estimator.hpp"
#include "../constants.hpp"

using namespace std;
using namespace cv;

class LO_RANSAC_Estimator : public FundMatEstimator{
	public:
		LO_RANSAC_Estimator(double param1, double param2, double param3){
			reprojectError = param1;
			confidence = param2;
			inlier_ratio = param3;
			INNER_LOOP = 30;
		}
		int run(_InputArray _points1, _InputArray _points2, _OutputArray _F, _OutputArray _mask = _OutputArray(), vector<float> similarities = vector<float>()){

			//Reset number of iterations and runtime
			num_iters = 0;
			loop_runtime = 0;

			Mat points1 = _points1.getMat(), points2 = _points2.getMat();
			Mat mask, model, model2, bestModel, subselect1, subselect2, bestMask1, bestMask2, subtest1, subtest2, subselect11, subselect12;

	        //Number of iterations set according to the standard termination criterion
	        int iter, niters = (int)ceil(log(1 - confidence)/log(1 - pow(inlier_ratio,8)));
	        int d1 = points1.channels() > 1 ? points1.channels() : points1.cols;
	        int d2 = points2.channels() > 1 ? points2.channels() : points2.cols;
	        int count = points1.checkVector(d1), count2 = points2.checkVector(d2), maxGoodCount = 0;

	        CV_Assert( confidence > 0 && confidence < 1 );
	        CV_Assert( count >= 0 && count2 == count );
	        if( count < MIN_MODEL_POINTS )
	            return 0;

	        if( _mask.needed() )
	        {
	            _mask.create(count, 1, CV_8U, -1, true);
	            bestMask2 = bestMask1 = _mask.getMat();
	            CV_Assert( (bestMask1.cols == 1 || bestMask1.rows == 1) && (int)bestMask1.total() == count );
	        }
	        else
	        {
	            bestMask1.create(count, 1, CV_8U);
	            bestMask2 = bestMask1;
	        }

	        if( count == MIN_MODEL_POINTS )
	        {
	            Estimator::fundMat(points1, points2, bestModel, true);
	            bestModel.copyTo(_F);
	            bestMask1.setTo(Scalar::all(1));
	            return MIN_MODEL_POINTS;
	        }

	        bool multichan = points1.channels() > 1;

	        clock_t t1,t2;
		    t1=clock();
		    
	        for( iter = 0; iter < niters; iter++ )
	        {
	        	//For benchmarking
	        	num_iters++;

	        	Estimator::subselect(points1, points2, subselect1, subselect2, MIN_MODEL_POINTS);
	            int i, nmodels;

                nmodels = Estimator::fundMat(subselect1, subselect2, model, true);
	            if( nmodels <= 0 )
	                continue;
	            CV_Assert( model.rows % nmodels == 0 );
	            Size modelSize(model.cols, model.rows/nmodels);

	            for( i = 0; i < nmodels; i++ )
	            {
	                Mat model_i = model.rowRange( i*modelSize.height, (i+1)*modelSize.height );
	                int goodCount = Estimator::getInliers( points1, points2, model_i, reprojectError, mask );

	                if( goodCount > max(maxGoodCount, MIN_MODEL_POINTS-1) )
	                {
	                    swap(mask, bestMask1);
	                    model_i.copyTo(bestModel);
	                    maxGoodCount = goodCount;

	                    if(multichan){
							subtest1.create(maxGoodCount, 1, points1.type());
							subtest2.create(maxGoodCount, 1, points2.type());
						} else {
							subtest1.create(maxGoodCount, 2, points1.type());
							subtest2.create(maxGoodCount, 2, points2.type());
						}
						int h = 0;
						for(int k = 0; k < count && h < maxGoodCount; k++){
							if(bestMask1.at<uchar>(k)){
								points1.row(k).copyTo(subtest1.row(h));
								points2.row(k).copyTo(subtest2.row(h));
								h++;
							}
						}

	                    for(int j = 0; j < INNER_LOOP; j++){

	                    	Estimator::subselect(subtest1, subtest2, subselect11, subselect12, MIN_MODEL_POINTS);
	                    	int n2models = Estimator::fundMat(subselect11, subselect12, model2, true);
	                    	if(n2models < 1)
	                    		break;
	                    	CV_Assert( model2.rows % n2models == 0 );
	            			Size model2Size(model2.cols, model2.rows/n2models);
	            			for(int l = 0; l < n2models; l++ ){
	            				Mat model_l = model2.rowRange( i*model2Size.height, (i+1)*model2Size.height );
	            				int betterCount = Estimator::getInliers( points1, points2, model_l, reprojectError, mask );
	            				if( betterCount > max(maxGoodCount, MIN_MODEL_POINTS-1) ){
	            					swap(mask, bestMask1);
	            					model_l.copyTo(bestModel);
	                    			maxGoodCount = betterCount;
	            				}
	            			}

	                    }
	                    niters = Estimator::updateNumIters( confidence, (double)(count - maxGoodCount)/count, MIN_MODEL_POINTS, niters );
	                }
				}
	            
	        }
	        t2=clock();
		    loop_runtime = ((float)t2-(float)t1)/CLOCKS_PER_SEC;

	        if( maxGoodCount > 0 )
	        {
	            if( bestMask1.data != bestMask2.data )
	            {
	                if( bestMask1.size() == bestMask2.size() )
	                    bestMask1.copyTo(bestMask2);
	                else
	                    transpose(bestMask1, bestMask2);
	            }
	            bestModel.copyTo(_F);
	        }
	        else
	            _F.release();

			return maxGoodCount;

			return 0;
		}
};