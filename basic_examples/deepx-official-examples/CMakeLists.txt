cmake_minimum_required(VERSION 3.14)

if(MSVC)
    if(NOT DEFINED DXRT_DIR OR NOT EXISTS ${DXRT_DIR})
        message(FATAL_ERROR "The variable DXRT_DIR is not set. The build cannot proceed.")
    endif()
    string(REPLACE "\\" "/" DXRT_DIR ${DXRT_DIR})
    set(DXRT_INSTALLED_DIR ${DXRT_DIR})
    if(NOT DEFINED OpenCV_DIR OR NOT EXISTS ${OpenCV_DIR})
        message(FATAL_ERROR "The variable OpenCV_DIR is not set. The build cannot proceed.")
    endif()
    string(REPLACE "\\" "/" OpenCV_DIR ${OpenCV_DIR})
endif()

project(dx_demos DESCRIPTION "DEEPX Demos")

get_filename_component(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR} ABSOLUTE)
add_compile_definitions(PROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/release" CACHE PATH "Default install path" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if(NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "${CMAKE_SYSTEM_PROCESSOR}")
    set(CROSS_COMPILE TRUE)
else()
    set(CROSS_COMPILE FALSE)
endif()

include(dxapp.cfg)
include(dxapp.function)

if(CROSS_COMPILE)
    set(CMAKE_SKIP_RPATH TRUE)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -MP")
    add_compile_options(-D_AMD64_ -D_WIN64 -DWIN64)
    add_compile_options(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)
else()
    add_compile_options(-W -Wall -pthread -fstrict-volatile-bitfields -fPIC)
endif()

add_subdirectory(demos/object_detection)
add_subdirectory(demos/classification)
add_subdirectory(demos/pose_estimation)
add_subdirectory(demos/object_det_and_seg)
add_subdirectory(demos/segmentation)