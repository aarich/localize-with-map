# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/robotics/Desktop/Localize/CreateDB

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/robotics/Desktop/Localize/CreateDB/build

# Include any dependencies generated for this target.
include CMakeFiles/createDBWithImages.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/createDBWithImages.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/createDBWithImages.dir/flags.make

CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o: CMakeFiles/createDBWithImages.dir/flags.make
CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o: ../src/createDBWithImages.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/robotics/Desktop/Localize/CreateDB/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o -c /home/robotics/Desktop/Localize/CreateDB/src/createDBWithImages.cpp

CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/robotics/Desktop/Localize/CreateDB/src/createDBWithImages.cpp > CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.i

CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/robotics/Desktop/Localize/CreateDB/src/createDBWithImages.cpp -o CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.s

CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o.requires:
.PHONY : CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o.requires

CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o.provides: CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o.requires
	$(MAKE) -f CMakeFiles/createDBWithImages.dir/build.make CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o.provides.build
.PHONY : CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o.provides

CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o.provides.build: CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o

# Object files for target createDBWithImages
createDBWithImages_OBJECTS = \
"CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o"

# External object files for target createDBWithImages
createDBWithImages_EXTERNAL_OBJECTS =

createDBWithImages: CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o
createDBWithImages: /usr/lib/libboost_system-mt.so
createDBWithImages: /usr/lib/libboost_filesystem-mt.so
createDBWithImages: /usr/lib/libboost_thread-mt.so
createDBWithImages: /usr/lib/libboost_date_time-mt.so
createDBWithImages: /usr/lib/libboost_iostreams-mt.so
createDBWithImages: /usr/lib/libboost_serialization-mt.so
createDBWithImages: /usr/lib/libpcl_common.so
createDBWithImages: /usr/lib/libflann_cpp_s.a
createDBWithImages: /usr/lib/libpcl_kdtree.so
createDBWithImages: /usr/lib/libpcl_octree.so
createDBWithImages: /usr/lib/libpcl_search.so
createDBWithImages: /usr/lib/libOpenNI.so
createDBWithImages: /usr/lib/libvtkCommon.so.5.8.0
createDBWithImages: /usr/lib/libvtkRendering.so.5.8.0
createDBWithImages: /usr/lib/libvtkHybrid.so.5.8.0
createDBWithImages: /usr/lib/libvtkCharts.so.5.8.0
createDBWithImages: /usr/lib/libpcl_io.so
createDBWithImages: /usr/lib/libpcl_sample_consensus.so
createDBWithImages: /usr/lib/libpcl_filters.so
createDBWithImages: /usr/lib/libpcl_visualization.so
createDBWithImages: /usr/lib/libpcl_outofcore.so
createDBWithImages: /usr/lib/libpcl_features.so
createDBWithImages: /usr/lib/libpcl_segmentation.so
createDBWithImages: /usr/lib/libpcl_people.so
createDBWithImages: /usr/lib/libpcl_registration.so
createDBWithImages: /usr/lib/libpcl_recognition.so
createDBWithImages: /usr/lib/libpcl_keypoints.so
createDBWithImages: /usr/lib/libqhull.so
createDBWithImages: /usr/lib/libpcl_surface.so
createDBWithImages: /usr/lib/libpcl_tracking.so
createDBWithImages: /usr/lib/libpcl_apps.so
createDBWithImages: /usr/lib/libboost_system-mt.so
createDBWithImages: /usr/lib/libboost_filesystem-mt.so
createDBWithImages: /usr/lib/libboost_thread-mt.so
createDBWithImages: /usr/lib/libboost_date_time-mt.so
createDBWithImages: /usr/lib/libboost_iostreams-mt.so
createDBWithImages: /usr/lib/libboost_serialization-mt.so
createDBWithImages: /usr/lib/libqhull.so
createDBWithImages: /usr/lib/libOpenNI.so
createDBWithImages: /usr/lib/libflann_cpp_s.a
createDBWithImages: /usr/lib/libvtkCommon.so.5.8.0
createDBWithImages: /usr/lib/libvtkRendering.so.5.8.0
createDBWithImages: /usr/lib/libvtkHybrid.so.5.8.0
createDBWithImages: /usr/lib/libvtkCharts.so.5.8.0
createDBWithImages: /opt/ros/hydro/lib/libopencv_videostab.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_video.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_ts.a
createDBWithImages: /opt/ros/hydro/lib/libopencv_superres.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_stitching.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_photo.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_ocl.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_objdetect.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_nonfree.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_ml.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_legacy.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_imgproc.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_highgui.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_gpu.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_flann.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_features2d.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_core.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_contrib.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_calib3d.so.2.4.9
createDBWithImages: /usr/lib/libpcl_common.so
createDBWithImages: /usr/lib/libpcl_kdtree.so
createDBWithImages: /usr/lib/libpcl_octree.so
createDBWithImages: /usr/lib/libpcl_search.so
createDBWithImages: /usr/lib/libpcl_io.so
createDBWithImages: /usr/lib/libpcl_sample_consensus.so
createDBWithImages: /usr/lib/libpcl_filters.so
createDBWithImages: /usr/lib/libpcl_visualization.so
createDBWithImages: /usr/lib/libpcl_outofcore.so
createDBWithImages: /usr/lib/libpcl_features.so
createDBWithImages: /usr/lib/libpcl_segmentation.so
createDBWithImages: /usr/lib/libpcl_people.so
createDBWithImages: /usr/lib/libpcl_registration.so
createDBWithImages: /usr/lib/libpcl_recognition.so
createDBWithImages: /usr/lib/libpcl_keypoints.so
createDBWithImages: /usr/lib/libpcl_surface.so
createDBWithImages: /usr/lib/libpcl_tracking.so
createDBWithImages: /usr/lib/libpcl_apps.so
createDBWithImages: /usr/lib/libvtkViews.so.5.8.0
createDBWithImages: /usr/lib/libvtkInfovis.so.5.8.0
createDBWithImages: /usr/lib/libvtkWidgets.so.5.8.0
createDBWithImages: /usr/lib/libvtkHybrid.so.5.8.0
createDBWithImages: /usr/lib/libvtkParallel.so.5.8.0
createDBWithImages: /usr/lib/libvtkVolumeRendering.so.5.8.0
createDBWithImages: /usr/lib/libvtkRendering.so.5.8.0
createDBWithImages: /usr/lib/libvtkGraphics.so.5.8.0
createDBWithImages: /usr/lib/libvtkImaging.so.5.8.0
createDBWithImages: /usr/lib/libvtkIO.so.5.8.0
createDBWithImages: /usr/lib/libvtkFiltering.so.5.8.0
createDBWithImages: /usr/lib/libvtkCommon.so.5.8.0
createDBWithImages: /usr/lib/libvtksys.so.5.8.0
createDBWithImages: /opt/ros/hydro/lib/libopencv_nonfree.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_ocl.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_gpu.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_photo.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_objdetect.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_legacy.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_video.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_ml.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_calib3d.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_features2d.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_highgui.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_imgproc.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_flann.so.2.4.9
createDBWithImages: /opt/ros/hydro/lib/libopencv_core.so.2.4.9
createDBWithImages: CMakeFiles/createDBWithImages.dir/build.make
createDBWithImages: CMakeFiles/createDBWithImages.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable createDBWithImages"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/createDBWithImages.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/createDBWithImages.dir/build: createDBWithImages
.PHONY : CMakeFiles/createDBWithImages.dir/build

CMakeFiles/createDBWithImages.dir/requires: CMakeFiles/createDBWithImages.dir/src/createDBWithImages.cpp.o.requires
.PHONY : CMakeFiles/createDBWithImages.dir/requires

CMakeFiles/createDBWithImages.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/createDBWithImages.dir/cmake_clean.cmake
.PHONY : CMakeFiles/createDBWithImages.dir/clean

CMakeFiles/createDBWithImages.dir/depend:
	cd /home/robotics/Desktop/Localize/CreateDB/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/robotics/Desktop/Localize/CreateDB /home/robotics/Desktop/Localize/CreateDB /home/robotics/Desktop/Localize/CreateDB/build /home/robotics/Desktop/Localize/CreateDB/build /home/robotics/Desktop/Localize/CreateDB/build/CMakeFiles/createDBWithImages.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/createDBWithImages.dir/depend

