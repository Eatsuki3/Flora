# Installation

## Environment

The code is developed and tested on **Ubuntu 20.04/22.04** with **GCC ≥ 9**
(C++14). The GPU backend additionally requires **CUDA ≥ 11.1** (see `computing-server/CMakeLists.txt` for the fallback).

> The EMP toolkit sources are **vendored** inside this repository
> (`emp-tool/`, `emp-ot/`, `emp-sh2pc/`) and the `emp-tool` multi-threading
> patch is **already applied**. You only need to build and install them.

## 1. System packages

```bash
sudo apt update
sudo apt install -y g++ cmake build-essential libgmp-dev libboost-all-dev \
                    libssl-dev python3-pip
```

Python dependencies (only needed for `dataset-scripts/` and optional traffic
analysis in `micro-eval/scripts/`):

```bash
pip3 install gdown absl-py pandas numpy matplotlib
```

## 2. Build and install the vendored EMP libraries

Install EMP to a local prefix (recommended; the example uses
`$HOME/flora-local`). `emp-tool` must be installed first because `emp-ot` and
`emp-sh2pc` depend on it.

```bash
# from the repository root
EMP_PREFIX=$HOME/flora-local

# emp-tool
pushd emp-tool
cmake -B build -DCMAKE_INSTALL_PREFIX=$EMP_PREFIX \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build
popd

# emp-ot
pushd emp-ot
cmake -B build -DCMAKE_INSTALL_PREFIX=$EMP_PREFIX \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build
popd

# emp-sh2pc
pushd emp-sh2pc
cmake -B build -DCMAKE_INSTALL_PREFIX=$EMP_PREFIX \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build
popd
```

Export the prefix so CMake can find the EMP config files:

```bash
export CMAKE_PREFIX_PATH=$EMP_PREFIX:$CMAKE_PREFIX_PATH
# add the line above to ~/.bashrc to make it persistent
```

> If you prefer a system-wide install, omit `-DCMAKE_INSTALL_PREFIX` (defaults
> to `/usr/local`) and run `cmake --install build` with `sudo`.

## 3. CUDA / GPU (optional)

For the `server_gpu` target, install the NVIDIA driver and CUDA Toolkit
(≥ 11.1). CMake detects `nvcc` automatically; if needed, point it at the
compiler explicitly:

```bash
cd computing-server
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

To build CPU-only:

```bash
cmake -B build -DBUILD_GPU=OFF
```

## 4. Verify

```bash
cd computing-server
cmake -B build
cmake --build build -j
ls build/bin/   # server  (and server_gpu if CUDA was found)
```

See [`README.md`](./README.md) for preprocessing and running.
