!#/bin/bash

apt-get update
apt-get install -y vim man wget unzip curl gnupg2 ca-certificates lsb-release apache2-utils ethtool wget build-essential zlib1g cmake pkg-config libglvnd-dev libegl1 libopenblas-dev liblapack-dev linux-headers-generic

# Install Nvidia Driver
apt install nvidia-cuda-toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run
sh cuda_12.0.1_525.85.12_linux.run --silent --toolkit

apt install gcc-10 g++-10

# Change default gcc and g++ version and cuda path
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDA_ROOT=/usr/local/cuda-12.0/
ln -s /usr/bin/gcc-10 $CUDA_ROOT/bin/gcc
ln -s /usr/bin/g++-10 $CUDA_ROOT/bin/g++

git clone https://github.com/rifkybujana/marian.git
cd marian/
git checkout patch-1
mkdir build
cd build
cmake ..
make -j4

cd ../..
git clone https://github.com/moses-smt/mosesdecoder.git

# Download model
wget https://object.pouta.csc.fi/OPUS-MT-models/en-id/opus-2019-12-18.zip
unzip opus-2019-12-18.zip -d ./model