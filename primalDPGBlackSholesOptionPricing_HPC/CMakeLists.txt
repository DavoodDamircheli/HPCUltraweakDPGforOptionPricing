cmake_minimum_required(VERSION 3.14)
project(primalDPG_HPC)

set(CMAKE_CXX_STANDARD 14)
#find_package (Eigen)

set(eigen_include_dir "eigen_include_dir" CACHE STRING "Directory containing eigen header files" )
set(mpi_include_dir "mpi_include_dir" CACHE STRING "Directory to Openmpi header files")
set(mpi_lib_dir "mpi_lib_dir" CACHE STRING "Directory containing MPI library")
#---------------------------- added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -fopenmp")

#-----------------------------PETSC cached variables ---------------------
set(petsc_include_dir "petsc_include_dir" CACHE STRING "Directory containing header flies")
set(petsc_lib_dir "petsc_lib_dir" CACHE STRING "Directory contain PETSC library fils")


find_library(mpi_lib mpi ${mpi_lib_dir})
find_library(petsc_lib petsc ${petsc_lib_dir})

#include_directories(${eigen_include_dir})
include_directories(${eigen_include_dir}
        ${mpi_include_dir}
        ${petsc_include_dir})


add_executable(primalDPG_HPC
        main.cpp
        Eigen
        cmake-build-debug
        FEValues.h
        primalEuropeanOptions.cpp
        primalEuropeanOptions.h
        BinomialHighamOptionPricing.cpp
        BinomialHighamOptionPricing.h)
#--------------- this should be after add_executable!!! ---------
target_link_libraries(primalDPG_HPC
        ${mpi_lib}
        ${mpi_lib_dir}/libmpi.dylib
        ${mpicxx_lib}
        ${petsc_lib})




if(MPI_COMPILE_FLAGS)
    set_target_properties(primalDPG_HPC PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(primalDPG_HPC PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()