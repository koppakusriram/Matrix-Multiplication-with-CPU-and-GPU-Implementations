
all: app0 app1 app2 app3 app4

app0:
	g++ matrixMul_cpu.cpp -o matrixMul_cpu.exe

app1:
	nvcc matrixMul_gpu.cu -o matrixMul_gpu.exe
	
app2:
	nvcc matrixMul_gpu_v2.cu -o matrixMul_gpu_v2.exe

app3:
	nvcc matrixMul_gpu_v3.cu -o matrixMul_gpu_v3.exe

app4:
	nvcc matrixMul_gpu_v4.cu -o matrixMul_gpu_v4.exe

clean:
	rm -rf *.exe

output: output_cpu output_gpu_1 output_gpu_2 output_gpu_3 output_gpu_4

output_cpu:
	./matrixMul_cpu.exe

output_gpu_1:
	./matrixMul_gpu.exe
	nvprof ./matrixMul_gpu.exe

output_gpu_2:
	./matrixMul_gpu_v2.exe
	nvprof ./matrixMul_gpu_v2.exe

output_gpu_3:
	./matrixMul_gpu_v3.exe
	nvprof ./matrixMul_gpu_v3.exe
	
output_gpu_4:
	./matrixMul_gpu_v4.exe
	nvprof ./matrixMul_gpu_v4.exe
