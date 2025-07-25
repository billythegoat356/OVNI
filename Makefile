ifndef CUDA_ARCH
$(error You must specify CUDA_ARCH. Check it here: https://developer.nvidia.com/cuda-gpus. Example: `make CUDA_ARCH=compute_89`)
endif

SRC = ovni/kernels/src/all_kernels.cu
OUT = ovni/kernels/compiled/all_kernels.ptx

all:
	nvcc -arch=$(CUDA_ARCH) -ptx $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)
