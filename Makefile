CUDA_ARCH = compute_89 # Write down your version from https://developer.nvidia.com/cuda-gpus

SRC = ovni/kernels/src/all_kernels.cu
OUT = ovni/kernels/compiled/all_kernels.ptx

all:
	nvcc -arch=$(CUDA_ARCH) -ptx $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)
