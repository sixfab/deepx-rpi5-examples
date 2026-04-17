cmake_minimum_required(VERSION 3.14)
project(DeepXCppExamples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------- dependencies
find_package(OpenCV REQUIRED)
find_library(DXRT_LIB NAMES dxrt REQUIRED)

option(ENABLE_LIBCAMERA "Enable Raspberry Pi libcamera support" ON)
if(ENABLE_LIBCAMERA)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(LIBCAMERA libcamera)
    endif()
endif()

# nlohmann/json via FetchContent. If it is already available on the system
# (e.g. `libnlohmann-json3-dev` package), use that instead to avoid a download.
find_package(nlohmann_json 3.7 QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE
    )
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install    OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ------------------------------------------------------------- shared library
file(GLOB COMMON_SOURCES common/*.cpp)
add_library(deepx_common STATIC ${COMMON_SOURCES})
target_include_directories(deepx_common PUBLIC common)
target_link_libraries(deepx_common
    PUBLIC
        ${OpenCV_LIBS}
        ${DXRT_LIB}
        nlohmann_json::nlohmann_json
)

if(LIBCAMERA_FOUND)
    target_compile_definitions(deepx_common PUBLIC ENABLE_LIBCAMERA)
    target_link_libraries(deepx_common PUBLIC ${LIBCAMERA_LIBRARIES})
    target_include_directories(deepx_common PUBLIC ${LIBCAMERA_INCLUDE_DIRS})
    message(STATUS "libcamera: enabled (${LIBCAMERA_VERSION})")
else()
    message(STATUS "libcamera: disabled")
endif()

# -------------------------------------------------------------- demo helper
# Used by every demo's CMakeLists.txt: `add_demo(name source.cpp)`.
macro(add_demo TARGET SOURCE)
    add_executable(${TARGET} ${SOURCE})
    target_link_libraries(${TARGET} PRIVATE deepx_common)
endmacro()

# ------------------------------------------------------------ demo categories
add_subdirectory(object_detection)
add_subdirectory(pose_estimation)
add_subdirectory(face)
add_subdirectory(ppu)
add_subdirectory(async_example)
add_subdirectory(advanced)
add_subdirectory(segmentation)
add_subdirectory(classification)
