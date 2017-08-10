// definitions for RANSAC variations
#define SFM_RANSAC 1
#define SFM_RANSAC_Tdd 2
#define SFM_PROSAC 3
#define SFM_PROSAC_Tdd 4
#define SFM_PE_RANSAC 5
#define SFM_LO_RANSAC 6

//By default we set the inlier ratio for RANSAC algorithms to 50% (50% outlier contamination)
#define INLIER_RATIO 0.5

//Minimum required points for 8-point linear equation solver
#define MIN_MODEL_POINTS 8

//Include only once, to avoid redefinitions
#pragma once