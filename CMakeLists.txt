# 1. 声明要求的cmake最低版本
cmake_minimum_required(VERSION 3.5)

# 3. 声明一个cmake工程
project(jet)

# set( CMAKE_CXX_FLAGS "-std=c++20" )

find_package(Ceres REQUIRED)

# helloworld
add_executable(jet jet.cc)
# 这个可执行文件所需的库（一般就是刚刚生成的工程的库咯）
target_link_libraries(jet Ceres::ceres)