cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
project( SFM )
find_package( OpenCV REQUIRED )

add_definitions(-D VERBOSE)

set( HEADER_FILES SFM/Image.hpp SFM/RANSAC/ransac.hpp SFM/ )
set( EXECUTABLE_FILES SFM/Image.cpp SFM/RANSAC/ransac.cpp )

add_executable( StructureFromMotion ${EXECUTABLE_FILES} ${HEADER_FILES} SFM.cpp )
target_link_libraries( StructureFromMotion ${OpenCV_LIBS} )
