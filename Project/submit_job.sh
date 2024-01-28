module load gcc cuda

nvcc matrixMul_gpu.cu -o matrixMul_gpu.exe
nvprof ./matrixMul_gpu.exe
