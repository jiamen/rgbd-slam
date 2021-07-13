
# 设定cmake版本
CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

# 设定工程名称
PROJECT( slam )
# 指定编译器
SET(CMAKE_CXX_COMPILER "g++")
# 指定编译版本
SET( CMAKE_BUILD_TYPE Debug )

SET( CMAKE_CXX_FLAGS "-O3 -std=c++14" )

# 设定可执行二进制文件的目录
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# 设定存放编译出来的库文件的目录
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# 设定头文件目录
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include )

# 并且把该目录设为链接目录
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)

# 增加子文件夹，也就是进入源代码文件夹继续构建
# ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src )



FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

# 增加opencv的依赖
FIND_PACKAGE( OpenCV REQUIRED )

# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

# 添加g2o的依赖
# 因为g2o不是常用库，要添加它的findg2o.cmake文件
LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O REQUIRED )
# CSparse
FIND_PACKAGE( CSparse REQUIRED )
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )




ADD_LIBRARY( slambase src/slamBase.cpp )
TARGET_LINK_LIBRARIES( slambase
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES} )

ADD_EXECUTABLE( slam src/slam.cpp )
TARGET_LINK_LIBRARIES( slam
        slambase
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})


