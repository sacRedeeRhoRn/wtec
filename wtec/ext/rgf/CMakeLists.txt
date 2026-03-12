cmake_minimum_required(VERSION 3.18)
project(wtec_rgf C)

find_package(MPI REQUIRED)
find_package(OpenMP)

file(GLOB WTEC_RGF_SOURCES CONFIGURE_DEPENDS src/*.c)

add_executable(wtec_rgf_runner ${WTEC_RGF_SOURCES})
target_include_directories(wtec_rgf_runner PRIVATE include)
target_link_libraries(wtec_rgf_runner PRIVATE MPI::MPI_C)
if(OpenMP_C_FOUND)
  target_link_libraries(wtec_rgf_runner PRIVATE OpenMP::OpenMP_C)
endif()
target_compile_features(wtec_rgf_runner PRIVATE c_std_99)
set(WTEC_RGF_BLAS_BACKEND "none" CACHE STRING "Reported BLAS backend for probe metadata")
if(NOT WTEC_RGF_BLAS_BACKEND STREQUAL "none")
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  target_link_libraries(wtec_rgf_runner PRIVATE BLAS::BLAS LAPACK::LAPACK)
  target_compile_definitions(wtec_rgf_runner PRIVATE WTEC_RGF_USE_LAPACK=1)
endif()
target_compile_definitions(
  wtec_rgf_runner
  PRIVATE
  WTEC_RGF_BUILD_BLAS_BACKEND="${WTEC_RGF_BLAS_BACKEND}"
)
