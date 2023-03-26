# Cache-optimized and low-overhead implementations of multigrid smoothers for high-order FEM computations

This project contains benchmarks for smoothers in the context of multigrid.

## Getting started

Build deal.II with p4est and Trilinos enabled (please specify the paths to those libraries):

```
git clone https://github.com/dealii/dealii.git
mkdir dealii-build
cd dealii-build/
echo "cmake \
    -D DEAL_II_WITH_64BIT_INDICES=\"OFF\" \
    -D CMAKE_BUILD_TYPE=\"DebugRelease\" \
    -D CMAKE_CXX_COMPILER=\"mpic++\" \
    -D CMAKE_CXX_FLAGS=\"-march=native -Wno-array-bounds  -std=c++17\" \
    -D DEAL_II_CXX_FLAGS_RELEASE=\"-O3\" \
    -D CMAKE_C_COMPILER=\"mpicc\" \
    -D DEAL_II_WITH_MPI=\"ON\" \
    -D DEAL_II_WITH_P4EST=\"ON\" \
    -D MPIEXEC_PREFLAGS=\"-bind-to none\" \
    -D DEAL_II_WITH_LAPACK=\"ON\" \
    -D DEAL_II_WITH_HDF5=\"OFF\" \
    -D DEAL_II_FORCE_BUNDLED_BOOST=\"OFF\" \
    -D DEAL_II_COMPONENT_DOCUMENTATION=\"OFF\" \
    -D P4EST_DIR=PATH_TO_P4EST \
    -D DEAL_II_WITH_TRILINOS=\"ON\" \
    -D TRILINOS_DIR=PATH_TO_TRILINOS \
    -D DEAL_II_WITH_PETSC:BOOL=\"ON\" \
    -D PETSC_DIR=PATH_TO_PETSC \
    -D PETSC_ARCH=\"arch-linux2-c-opt\" \
    ../dealii" > config.sh
. config.sh
make -j30
cd ..
```

Build the benchmarks:
```
git clone https://github.com/peterrum/dealii-dd-and-schwarz.git
mkdir dealii-dd-and-schwarz-build
cd dealii-dd-and-schwarz-build
cmake ../dealii-multigrid -DDEAL_II_DIR=../dealii-build
make release
make -j10
cd ..
```

Run the experiments in the `experiments` folder.

Data collected during extensive parameter studies are given in the `data` folder.

