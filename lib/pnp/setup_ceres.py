import os
from cffi import FFI

if __name__ == "__main__":
    include_dir = os.getenv('CERES_INCLUDE_DIR','/usr/local/include')
    ceres_so_path = os.getenv('CERES_LIB_DIR','/usr/local/lib') + '/libceres.so'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    old_wd = os.getcwd()
    os.chdir(dir_path)

    os.system(
        'gcc -shared cxx/ceres.cpp -c '
        '-o cxx/ceres.cpp.o '
        '-fopenmp -fPIC -O2 -std=c++17 '
        '-I/usr/include -I/usr/include/eigen3' + f' -I{include_dir}')

    ffibuilder = FFI()
    with open("cxx/ext.h") as f:
        ffibuilder.cdef(f.read())
        # -lcudart -lcublas -lcusolver -lglog -lpthread -lcxsparse -lcholmod -llapack
    ffibuilder.set_source(
        "_ext",
        """
        #include "cxx/ext.h"
        """,
        extra_objects=['cxx/ceres.cpp.o', ceres_so_path],
        libraries=['stdc++', 'glog', 'pthread', 'cxsparse', 'cholmod','lapack'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
    ffibuilder.compile(verbose=True)
    os.system("rm cxx/ceres.cpp.o")
    os.system("rm _ext.c")
    os.system("rm _ext.o")
    os.chdir(old_wd)
    