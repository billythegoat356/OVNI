

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
void round_mask(
    unsigned char* dst,
    int radius_top_left,
    int radius_top_right,
    int radius_bottom_right,
    int radius_bottom_left,
    float smoothing,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = (y * width + x);


    float dx = 0.0f;
    float dy = 0.0f;
    float d;

    unsigned char alpha;

    // top-left
    if (x < radius_top_left && y < radius_top_left) {
        dx = (radius_top_left - 1) - x;
        dy = (radius_top_left - 1) - y;

        if (smoothing == 2.0f) {
            d = sqrtf(dx*dx + dy*dy);
        } else {
            d = powf(powf(fabsf(dx), smoothing) + powf(fabsf(dy), smoothing), 1.0f / smoothing);
        }
        float a = radius_top_left + 0.5f - d;
        a = fminf(fmaxf(a, 0.0f), 1.0f);
        alpha = (unsigned char)(a * 255.0f);

    }

    // top-right
    else if (x >= width - radius_top_right && y < radius_top_right) {
        dx = x - (width - radius_top_right);
        dy = (radius_top_right - 1) - y;

        if (smoothing == 2.0f) {
            d = sqrtf(dx*dx + dy*dy);
        } else {
            d = powf(powf(fabsf(dx), smoothing) + powf(fabsf(dy), smoothing), 1.0f / smoothing);
        }
        float a = radius_top_right + 0.5f - d;
        a = fminf(fmaxf(a, 0.0f), 1.0f);
        alpha = (unsigned char)(a * 255.0f);
    }

    // bottom-right
    else if (x >= width - radius_bottom_right && y >= height - radius_bottom_right) {
        dx = x - (width - radius_bottom_right);
        dy = y - (height - radius_bottom_right);

        if (smoothing == 2.0f) {
            d = sqrtf(dx*dx + dy*dy);
        } else {
            d = powf(powf(fabsf(dx), smoothing) + powf(fabsf(dy), smoothing), 1.0f / smoothing);
        }
        float a = radius_bottom_right + 0.5f - d;
        a = fminf(fmaxf(a, 0.0f), 1.0f);
        alpha = (unsigned char)(a * 255.0f);
    }

    // bottom-left
    else if (x < radius_bottom_left && y >= height - radius_bottom_left) {
        dx = (radius_bottom_left - 1) - x;
        dy = y - (height - radius_bottom_left);

        if (smoothing == 2.0f) {
            d = sqrtf(dx*dx + dy*dy);
        } else {
            d = powf(powf(fabsf(dx), smoothing) + powf(fabsf(dy), smoothing), 1.0f / smoothing);
        }
        float a = radius_bottom_left + 0.5f - d;
        a = fminf(fmaxf(a, 0.0f), 1.0f);
        alpha = (unsigned char)(a * 255.0f);
    }

    

    else {
        return;
    }


    dst[index] = (dst[index] * alpha + 127) / 255;
}


__device__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}


extern "C" __global__
void vignette(
    unsigned char* src,
    float strength,      // 0.0 = no effect, 1.0 = full black at edges
    float radius,        // 0.0 = vignette starts from center, 1.0 = starts on edges
    float softness,      // 0.0 -> 1.0, width of transition
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = (y * width + x) * 3;

    // Normalize coordinates to [-1, 1] centered at image middle
    float nx = (2.0f * x / width) - 1.0f;
    float ny = (2.0f * y / height) - 1.0f;

    // Distance from center
    float dist = sqrtf(nx * nx + ny * ny);
    dist *= 0.70710678f; // scale to 0-1

    // Calculate vignette factor
    float falloff = smoothstep(radius, radius + softness, dist);
    float factor = 1.0f - falloff * strength;

    // Apply to RGB channels
    src[index]     = (unsigned char)(src[index]     * factor);
    src[index + 1] = (unsigned char)(src[index + 1] * factor);
    src[index + 2] = (unsigned char)(src[index + 2] * factor);
}
