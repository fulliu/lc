mkdir /ceres-build
if [ $? -eq 0 ]; then
    echo ok
else
    echo failed to create /ceres-build
    exit 1
fi

cd /ceres-build
apt-get install cmake -y
# google-glog + gflags
apt-get install libgoogle-glog-dev libgflags-dev -y
# Use ATLAS for BLAS & LAPACK
apt-get install libatlas-base-dev -y
# Eigen3
apt-get install libeigen3-dev -y
apt-get install libsuitesparse-dev -y
wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
tar zxf ceres-solver-2.1.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake -DBUILD_SHARED_LIBS=ON -DCUDA=OFF ../ceres-solver-2.1.0 # important: -DBUILD_SHARED_LIBS=ON
make -j8
make install
