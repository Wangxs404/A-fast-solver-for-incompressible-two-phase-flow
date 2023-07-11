# CMakeList.txt: RB_Para 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project ("RB_Para")
set(CMAKE_CXX_STANDARD 17)

add_executable (RB_Para "main.cpp" "helpers.h")


