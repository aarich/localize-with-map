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
include CMakeFiles/createDBWithPC.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/createDBWithPC.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/createDBWithPC.dir/flags.make

CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o: CMakeFiles/createDBWithPC.dir/flags.make
CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o: ../src/createDBWithPointCloud.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/robotics/Desktop/Localize/CreateDB/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o -c /home/robotics/Desktop/Localize/CreateDB/src/createDBWithPointCloud.cpp

CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/robotics/Desktop/Localize/CreateDB/src/createDBWithPointCloud.cpp > CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.i

CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/robotics/Desktop/Localize/CreateDB/src/createDBWithPointCloud.cpp -o CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.s

CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o.requires:
.PHONY : CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o.requires

CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o.provides: CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o.requires
	$(MAKE) -f CMakeFiles/createDBWithPC.dir/build.make CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o.provides.build
.PHONY : CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o.provides

CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o.provides.build: CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o

# Object files for target createDBWithPC
createDBWithPC_OBJECTS = \
"CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o"

# External object files for target createDBWithPC
createDBWithPC_EXTERNAL_OBJECTS =

createDBWithPC: CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o
createDBWithPC: /usr/lib/libboost_system-mt.so
createDBWithPC: /usr/lib/libboost_filesystem-mt.so
createDBWithPC: /usr/lib/libboost_thread-mt.so
createDBWithPC: /usr/lib/libboost_date_time-mt.so
createDBWithPC: /usr/lib/libboost_iostreams-mt.so
createDBWithPC: /usr/lib/libboost_serialization-mt.so
createDBWithPC: /usr/lib/libpcl_common.so
createDBWithPC: /usr/lib/libflann_cpp_s.a
createDBWithPC: /usr/lib/libpcl_kdtree.so
createDBWithPC: /usr/lib/libpcl_octree.so
createDBWithPC: /usr/lib/libpcl_search.so
createDBWithPC: /usr/lib/libOpenNI.so
createDBWithPC: /usr/lib/libvtkCommon.so.5.8.0
createDBWithPC: /usr/lib/libvtkRendering.so.5.8.0
createDBWithPC: /usr/lib/libvtkHybrid.so.5.8.0
createDBWithPC: /usr/lib/libvtkCharts.so.5.8.0
createDBWithPC: /usr/lib/libpcl_io.so
createDBWithPC: /usr/lib/libpcl_sample_consensus.so
createDBWithPC: /usr/lib/libpcl_filters.so
createDBWithPC: /usr/lib/libpcl_visualization.so
createDBWithPC: /usr/lib/libpcl_outofcore.so
createDBWithPC: /usr/lib/libpcl_features.so
createDBWithPC: /usr/lib/libpcl_segmentation.so
createDBWithPC: /usr/lib/libpcl_people.so
createDBWithPC: /usr/lib/libpcl_registration.so
createDBWithPC: /usr/lib/libpcl_recognition.so
createDBWithPC: /usr/lib/libpcl_keypoints.so
createDBWithPC: /usr/lib/libqhull.so
createDBWithPC: /usr/lib/libpcl_surface.so
createDBWithPC: /usr/lib/libpcl_tracking.so
createDBWithPC: /usr/lib/libpcl_apps.so
createDBWithPC: /usr/lib/libboost_system-mt.so
createDBWithPC: /usr/lib/libboost_filesystem-mt.so
createDBWithPC: /usr/lib/libboost_thread-mt.so
createDBWithPC: /usr/lib/libboost_date_time-mt.so
createDBWithPC: /usr/lib/libboost_iostreams-mt.so
createDBWithPC: /usr/lib/libboost_serialization-mt.so
createDBWithPC: /usr/lib/libqhull.so
createDBWithPC: /usr/lib/libOpenNI.so
createDBWithPC: /usr/lib/libflann_cpp_s.a
createDBWithPC: /usr/lib/libvtkCommon.so.5.8.0
createDBWithPC: /usr/lib/libvtkRendering.so.5.8.0
createDBWithPC: /usr/lib/libvtkHybrid.so.5.8.0
createDBWithPC: /usr/lib/libvtkCharts.so.5.8.0
createDBWithPC: /opt/ros/hydro/lib/libopencv_videostab.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_video.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_ts.a
createDBWithPC: /opt/ros/hydro/lib/libopencv_superres.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_stitching.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_photo.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_ocl.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_objdetect.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_nonfree.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_ml.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_legacy.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_imgproc.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_highgui.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_gpu.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_flann.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_features2d.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_core.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_contrib.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_calib3d.so.2.4.9
createDBWithPC: /usr/lib/libpcl_common.so
createDBWithPC: /usr/lib/libpcl_kdtree.so
createDBWithPC: /usr/lib/libpcl_octree.so
createDBWithPC: /usr/lib/libpcl_search.so
createDBWithPC: /usr/lib/libpcl_io.so
createDBWithPC: /usr/lib/libpcl_sample_consensus.so
createDBWithPC: /usr/lib/libpcl_filters.so
createDBWithPC: /usr/lib/libpcl_visualization.so
createDBWithPC: /usr/lib/libpcl_outofcore.so
createDBWithPC: /usr/lib/libpcl_features.so
createDBWithPC: /usr/lib/libpcl_segmentation.so
createDBWithPC: /usr/lib/libpcl_people.so
createDBWithPC: /usr/lib/libpcl_registration.so
createDBWithPC: /usr/lib/libpcl_recognition.so
createDBWithPC: /usr/lib/libpcl_keypoints.so
createDBWithPC: /usr/lib/libpcl_surface.so
createDBWithPC: /usr/lib/libpcl_tracking.so
createDBWithPC: /usr/lib/libpcl_apps.so
createDBWithPC: /usr/lib/libvtkViews.so.5.8.0
createDBWithPC: /usr/lib/libvtkInfovis.so.5.8.0
createDBWithPC: /usr/lib/libvtkWidgets.so.5.8.0
createDBWithPC: /usr/lib/libvtkHybrid.so.5.8.0
createDBWithPC: /usr/lib/libvtkParallel.so.5.8.0
createDBWithPC: /usr/lib/libvtkVolumeRendering.so.5.8.0
createDBWithPC: /usr/lib/libvtkRendering.so.5.8.0
createDBWithPC: /usr/lib/libvtkGraphics.so.5.8.0
createDBWithPC: /usr/lib/libvtkImaging.so.5.8.0
createDBWithPC: /usr/lib/libvtkIO.so.5.8.0
createDBWithPC: /usr/lib/libvtkFiltering.so.5.8.0
createDBWithPC: /usr/lib/libvtkCommon.so.5.8.0
createDBWithPC: /usr/lib/libvtksys.so.5.8.0
createDBWithPC: /opt/ros/hydro/lib/libopencv_nonfree.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_ocl.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_gpu.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_photo.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_objdetect.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_legacy.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_video.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_ml.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_calib3d.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_features2d.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_highgui.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_imgproc.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_flann.so.2.4.9
createDBWithPC: /opt/ros/hydro/lib/libopencv_core.so.2.4.9
createDBWithPC: CMakeFiles/createDBWithPC.dir/build.make
createDBWithPC: CMakeFiles/createDBWithPC.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable createDBWithPC"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/createDBWithPC.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/createDBWithPC.dir/build: createDBWithPC
.PHONY : CMakeFiles/createDBWithPC.dir/build

CMakeFiles/createDBWithPC.dir/requires: CMakeFiles/createDBWithPC.dir/src/createDBWithPointCloud.cpp.o.requires
.PHONY : CMakeFiles/createDBWithPC.dir/requires

CMakeFiles/createDBWithPC.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/createDBWithPC.dir/cmake_clean.cmake
.PHONY : CMakeFiles/createDBWithPC.dir/clean

CMakeFiles/createDBWithPC.dir/depend:
	cd /home/robotics/Desktop/Localize/CreateDB/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/robotics/Desktop/Localize/CreateDB /home/robotics/Desktop/Localize/CreateDB /home/robotics/Desktop/Localize/CreateDB/build /home/robotics/Desktop/Localize/CreateDB/build /home/robotics/Desktop/Localize/CreateDB/build/CMakeFiles/createDBWithPC.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/createDBWithPC.dir/depend

