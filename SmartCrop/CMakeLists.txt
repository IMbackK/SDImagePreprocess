cmake_minimum_required(VERSION 3.6)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

set(SRC_FILES main.cpp yolo.cpp tokenize.cpp log.cpp seamcarving.cpp utils.cpp intelligentroi.cpp facerecognizer.cpp)

add_executable(smartcrop ${SRC_FILES})
target_link_libraries(smartcrop ${OpenCV_LIBS} -ltbb)
target_include_directories(smartcrop PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_options(smartcrop PRIVATE -s -g -Wall)
message(WARNING ${WEIGHT_DIR})
target_compile_definitions(smartcrop PUBLIC WEIGHT_DIR="${WEIGHT_DIR}")

install(TARGETS smartcrop RUNTIME DESTINATION bin)
