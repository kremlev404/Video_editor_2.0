cmake_minimum_required(VERSION 2.8)

project(video_editor_assistance)

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES ( "C:/Projects/video_editor_assistance/libs/Eigen3" )

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include
)



file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*)

source_group("include", FILES ${MAIN_HEADERS})


add_executable(${CMAKE_PROJECT_NAME} ${MAIN_SRC} application/main.cpp ${MAIN_HEADERS})
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_range_for)

target_link_libraries(${CMAKE_PROJECT_NAME}
  ${OpenCV_LIBS}
)