cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
 
project(localization)
 
find_package( PCL 1.7 REQUIRED )
find_package( OpenCV REQUIRED )

 
include_directories(${PCL_INCLUDE_DIRS})
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
 
set(PCL_BUILD_TYPE Release)
 
file(GLOB localization
    "src/*.h"
    "src/*.cpp"
)
add_executable(localization ${localization})
 
target_link_libraries (localization ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
