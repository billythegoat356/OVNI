

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



extern "C" __global__
void round_corners(
    unsigned char* src,
    int radius,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = (y * width + x) * 4;

    int alpha = 255;

    float dx = 0.0f;
    float dy = 0.0f;

    // top-left
    if (x < radius && y < radius) {
        dx = (radius - 1) - x;
        dy = (radius - 1) - y;
        
    }

    // top-right
    else if (x >= width - radius && y < radius) {
        dx = x - (width - radius);
        dy = (radius - 1) - y;
    }

    // bottom-left
    else if (x < radius && y >= height - radius) {
        dx = (radius - 1) - x;
        dy = y - (height - radius);
    }

    // bottom-right
    else if (x >= width - radius && y >= height - radius) {
        dx = x - (width - radius);
        dy = y - (height - radius);
    }

    else {
        return;
    }

    float d  = sqrtf(dx*dx + dy*dy);

    float a = radius + 0.5f - d;
    a = fminf(fmaxf(a, 0.0f), 1.0f);
    alpha = (unsigned char)(a * 255.0f);

    // Modulate existing alpha
    src[index + 3] = (src[index + 3] * alpha + 127) / 255;
}
