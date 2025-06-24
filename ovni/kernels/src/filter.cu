
__device__ inline float clampf(
    float value,
    float lower,
    float upper
) {
    return (value < lower) ? lower : (value > upper) ? upper : value;
}

extern "C" __global__
void chroma_key(
    unsigned char* src,
    unsigned char R,
    unsigned char G,
    unsigned char B,
    int minT,
    int maxT,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Define index
    int index = (y * width + x) * 4;

    // Get each channel from source
    unsigned char sR = src[index];
    unsigned char sG = src[index + 1];
    unsigned char sB = src[index + 2];

    // Compute difference with L2 norm
    float metric = sqrtf(powf(R-sR, 2) + powf(G-sG, 2) + powf(B-sB, 2));

    // Calculate alpha
    float alpha = (metric - minT) / (float)(maxT - minT);
    unsigned char n_alpha = (unsigned char)(clampf(alpha * 255, 0.0f, 255.0f));

    src[index + 3] = n_alpha;
}