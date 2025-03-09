#!/bin/bash

nvcc -O3 -std=c++17 -arch=sm_90 \
  -I/opt/hpcx/ompi/include \
  -L/opt/hpcx/ompi/lib \
  -lnuma -libverbs -lmpi \
  -lcuda -lcudart \
  nvls_test.cu -o nvls_test 


mpirun --bind-to numa -np 8 ./nvls_test > nvls_test.log 2>&1
