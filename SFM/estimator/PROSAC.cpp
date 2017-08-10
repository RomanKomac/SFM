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

class PROSAC_Estimator : public FundMatEstimator{
	public:
		PROSAC_Estimator(double param1, double param2, double param3){
			reprojectError = param1;
			confidence = param2;
			inlier_ratio = param3;

		}
		int run(_InputArray _points1, _InputArray _points2, _OutputArray _F, _OutputArray _mask = _OutputArray(), vector<float> similarities = vector<float>()){
			if(!similarities.size()){
					cout << "similarity scores required to run SFM_PROSAC estimator" << endl;
				CV_Assert( similarities.size() );
			}
			Mat points1 = _points1.getMat(), points2 = _points2.getMat();
			Mat mask, model, bestModel, subselect1, subselect2, bestMask1, bestMask2;

			//Reset number of iterations and runtime
			num_iters = 0;
			loop_runtime = 0;

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

	        currIter = 0;
	        stageIters = 1;
	        stageN = MIN_MODEL_POINTS;
	        Tn = niters;
			for (int i = 0; i < MIN_MODEL_POINTS; i++) {
				Tn *= MIN_MODEL_POINTS-i;
				Tn /= count-i;
			}

			clock_t t1,t2;
		    t1=clock();
	        for( iter = 0; iter < niters; iter++ )
	        {
	        	//For benchmarking
	        	num_iters++;

	        	currIter++;
				if (currIter >= stageIters && stageN < count) {
					stageN++;

					double Tn1 = Tn * (double)(stageN+1) / (stageN+1-MIN_MODEL_POINTS);
					double stageIts = Tn1 - Tn;
					stageIters = (int)ceil(stageIts);
					currIter = 0;
					Tn = Tn1;
				}

	        	Estimator::subselect(points1, points2, subselect1, subselect2, MIN_MODEL_POINTS, stageN);

	            int i, nmodels;

                nmodels = Estimator::fundMat(subselect1, subselect2, model, false); 
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
	                    niters = Estimator::updateNumIters( confidence, (double)(count - goodCount)/count, MIN_MODEL_POINTS, niters );
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
		}
	private:
		//Max number of samples drawn
		double Tn;
		//Current iterations
		int stageIters;
		int currIter;
		//Current stage number of sample points
		int stageN;
};