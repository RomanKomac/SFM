cmake_minimum_required(VERSION 2.8)
macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

project( SFM )
find_package( OpenCV REQUIRED )

add_definitions(-D VERBOSE)

set( HEADER_FILES SFM/Image.hpp SFM/SFM.hpp SFM/estimator/ransac.hpp SFM/ )
set( EXECUTABLE_FILES SFM/Image.cpp SFM/SFM.cpp SFM/estimator/ransac.cpp )

add_executable( StructureFromMotion ${EXECUTABLE_FILES} ${HEADER_FILES} tester.cpp )
target_link_libraries( StructureFromMotion ${OpenCV_LIBS} )
