cmake_minimum_required(VERSION 3.0)
project(model_cpp)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add source files
set(SOURCE_FILES main.cpp conv_layers.cpp file_loader.cpp layer_computer.cpp)

# Add executable based on the source files
add_executable(model_cpp ${SOURCE_FILES})

# Link OpenCV libraries
target_link_libraries(model_cpp ${OpenCV_LIBS})
