#################################################

project(separacaograosdecafe)
cmake_minimum_required(VERSION 2.8)

################################################################################
# COMPILER
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


################################################################################
# CMake variables
set(${PROJECT_NAME}_ROOT_DIR         ${CMAKE_SOURCE_DIR})

SET(CMAKE_MODULE_PATH                ${${PROJECT_NAME}_ROOT_DIR}/cmake)
SET(CMAKE_CURRENT_BINARY_DIR            ${${PROJECT_NAME}_ROOT_DIR}/build)
SET(CMAKE_INSTALL_PREFIX             ${${PROJECT_NAME}_ROOT_DIR}/bin)

get_filename_component(OpenCV_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)


################################################################################
# OpenCV

set(OpenCV_DIR "/home/chiruno/Desktop/Processamento Digital de Sinais/opencv-3.2.0/bin/share/OpenCV")	#Configurar para o diretorio da maquina.

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
	include_directories(${OpenCV_INCLUDE_DIRS})
	link_directories(${OpenCV_LIB_DIR})
endif()



################################################################################
# FILES
set(HDRS
    #src/<header>.hpp
)


set(SRCS
    src/main.cpp
)

################################################################################
# EXEC
add_executable(${PROJECT_NAME} ${HDRS} ${SRCS})


# LIBS
set(LIBS
    ${OpenCV_LIBS}
)

# LINK
target_link_libraries(${PROJECT_NAME} ${LIBS})


#Install
