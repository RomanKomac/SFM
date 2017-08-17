
cmake_minimum_required(VERSION 2.8)

#--------------------------------------------------------------------
#Build definitions
#--------------------------------------------------------------------
#Required cpp version 
macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)

  endif ()
endmacro(use_cxx11)

#Project name
project( SFM )

#Build mode (DEBUG, VERBOSE, RELEASE)
add_definitions(-D VERBOSE)
#--------------------------------------------------------------------


#--------------------------------------------------------------------
#Package definitions
#--------------------------------------------------------------------
#OpenCV library
find_package( OpenCV REQUIRED )

#Ceres solver library
find_package( Ceres REQUIRED )
include_directories(${CERES_INCLUDE_DIRS})
#--------------------------------------------------------------------


#--------------------------------------------------------------------
#File definitions
#--------------------------------------------------------------------
#Estimator and RANSAC implementations
file(GLOB ESTIMATOR_FILES
    "SFM/estimator/*.hpp"
    "SFM/estimator/*.cpp"
)

#Estimator and RANSAC implementations
file(GLOB BUNDLE_ADJUSTMENT_FILES
    "SFM/bundle_adjustment/*.hpp"
    "SFM/bundle_adjustment/*.cpp"
)

#Structure From Motion and Multiple View Stereo files
set( SFM_MVS_FILES SFM/SFM.hpp SFM/SFM.cpp SFM/MVS.hpp SFM/MVS.cpp )

#Helper files, such as loaders, readers, writers
set( HELPER_FILES SFM/Image.hpp SFM/Image.cpp )

#Predefined constants and definitions
set( PREDEF_FILES SFM/constants.hpp SFM/typedefs.hpp )
#---------------------------------------------------------------------


add_executable( StructureFromMotion ${SFM_MVS_FILES} ${HELPER_FILES} ${ESTIMATOR_FILES} ${BUNDLE_ADJUSTMENT_FILES} ${PREDEF_FILES} tester.cpp )
target_link_libraries( StructureFromMotion ${OpenCV_LIBS} ${CERES_LIBRARIES})