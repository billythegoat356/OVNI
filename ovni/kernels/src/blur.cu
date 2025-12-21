

extern "C" __global__
void gaussian_blur_horizontal(
    unsigned char* src,
    unsigned char* dst,
    float* weights,
    int radius,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int base = (y * width + x);

    float P = 0.f;

    for (int i = -radius; i <= radius; ++i) {
        int xx = min(max(x + i, 0), width - 1);
        int idx = (y * width + xx);
        float w = weights[i + radius];

        P += src[idx] * w;
    }

    dst[base] = (unsigned char)(P + 0.5f);
}



extern "C" __global__
void gaussian_blur_vertical(
    unsigned char* src,
    unsigned char* dst,
    float* weights,
    int radius,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int base = (y * width + x);

    float P = 0.f;

    for (int i = -radius; i <= radius; ++i) {
        int yy = min(max(y + i, 0), height - 1);
        int idx = (yy * width + x);
        float w = weights[i + radius];

        P += src[idx] * w;
    }

    dst[base] = (unsigned char)(P + 0.5f);
}


