//Include only once, to avoid redefinitions
#pragma once

//Definitions for RANSAC variations
#define SFM_RANSAC 1
#define SFM_RANSAC_Tdd 2
#define SFM_PROSAC 3
#define SFM_PROSAC_Tdd 4
#define SFM_PE_RANSAC 5
#define SFM_LO_RANSAC 6

//Definitions for matching
#define MATCH_EXHAUSTIVE 0
#define MATCH_CONSECUTIVE 1

//Definitions for reprojection cost function
#define BA_RE_SNAVELY 0
#define BA_RE_SIMPLE 1

//Definitions for dense reconstruction variations
#define DISPARITY 0
#define PATCH 1

//By default we set the inlier ratio for RANSAC algorithms to 50% (50% outlier contamination)
#define INLIER_RATIO 0.5

//Rule of thumb for mytch filtering proposed by Lowe
#define LOWE_MATCH_FILTER_RATIO 0.8f

//Minimal ratio when considering strong camera pair bundle adjustment
#define MINIMAL_BA_INLIER_RATIO 0.5

//Pixel reprojection error after triangulation
#define MIN_REPR_ERROR 4

//MIN_MATCH_DISTANCE
#define MERGE_CLOUD_POINT_MIN_MATCH_DISTANCE 0.02

//Minimum required points for 8-point linear equation solver
#define MIN_MODEL_POINTS 8

//Parameters for visualization
#define VIZ_CAM_PATH 1
#define VIZ_SPARSE 2
#define VIZ_BEFORE_BA 4