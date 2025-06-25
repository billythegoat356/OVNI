

extern "C" __global__
void chroma_key(
    unsigned char* src,
    unsigned char kR,
    unsigned char kG,
    unsigned char kB,
    int transparencyT,
    int opacityT,
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
    float metric = sqrtf(powf(kR-sR, 2) + powf(kG-sG, 2) + powf(kB-sB, 2));

    // Calculate alpha
    float alpha = (metric - transparencyT) / (float)(opacityT - transparencyT);
    unsigned char n_alpha = (unsigned char)(fmin(fmax(alpha * 255, 0.0f), 255.0f));

    src[index + 3] = n_alpha;
}