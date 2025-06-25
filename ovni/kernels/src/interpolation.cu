

// Applies bilinear interpolation at the given coordinates
__device__
void bilinear_pixel(
    const unsigned char* src,
    int width,
    int height,
    float x,
    float y,
    unsigned char *R,
    unsigned char *G,
    unsigned char *B
) {

    // If out of bounds, make it black
    if (x < 0 || x > width || y < 0 || y > height) {
        *R = 0;
        *G = 0;
        *B = 0;
        return;
    }

    // Clamp to source image bounds
    x = fmaxf(0.0f, fminf(x, width - 1.0f));
    y = fmaxf(0.0f, fminf(y, height - 1.0f));

    // Calculate surrounding bounds
    int left_x = int(floorf(x));
    int right_x = min(left_x + 1, width - 1);

    int top_y = int(floorf(y));
    int bottom_y = min(top_y + 1, height - 1);

    // Calculate X coordinates averages
    float x_cond = x - left_x;

    int flattened_top_left = (top_y * width + left_x) * 3;
    int flattened_top_right = (top_y * width + right_x) * 3;

    float R_top = src[flattened_top_left] * (1 - x_cond) + src[flattened_top_right] * x_cond;
    float G_top = src[flattened_top_left + 1] * (1 - x_cond) + src[flattened_top_right + 1] * x_cond;
    float B_top = src[flattened_top_left + 2] * (1 - x_cond) + src[flattened_top_right + 2] * x_cond;

    int flattened_bottom_left = (bottom_y * width + left_x) * 3;
    int flattened_bottom_right = (bottom_y * width + right_x) * 3;

    float R_bottom = src[flattened_bottom_left] * (1 - x_cond) + src[flattened_bottom_right] * x_cond;
    float G_bottom = src[flattened_bottom_left + 1] * (1 - x_cond) + src[flattened_bottom_right + 1] * x_cond;
    float B_bottom = src[flattened_bottom_left + 2] * (1 - x_cond) + src[flattened_bottom_right + 2] * x_cond;

    // Calculate Y coordinates average
    float y_cond = y - top_y;

    *R = roundf(R_top * (1 - y_cond) + R_bottom * y_cond);
    *G = roundf(G_top * (1 - y_cond) + G_bottom * y_cond);
    *B = roundf(B_top * (1 - y_cond) + B_bottom * y_cond);

}


// Applies bilinear blending between 4 images
extern "C" __global__
void bilinear_blend(
    unsigned char* dst,
    const unsigned char* top_left,
    const unsigned char* top_right,
    const unsigned char* bottom_left,
    const unsigned char* bottom_right,
    int width,
    int height,
    float x_distance,
    float y_distance
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int coords = (y * width + x) * 3;

    // Calculate X values
    float R = top_left[coords] * (1 - x_distance) + top_right[coords] * x_distance;
    float G = top_left[coords + 1] * (1 - x_distance) + top_right[coords + 1] * x_distance;
    float B = top_left[coords + 2] * (1 - x_distance) + top_right[coords + 2] * x_distance;

    float R2 = bottom_left[coords] * (1 - x_distance) + bottom_right[coords] * x_distance;
    float G2 = bottom_left[coords + 1] * (1 - x_distance) + bottom_right[coords + 1] * x_distance;
    float B2 = bottom_left[coords + 2] * (1 - x_distance) + bottom_right[coords + 2] * x_distance;

    // Now calculate Y
    R = R * (1 - y_distance) + R2 * y_distance;
    G = G * (1 - y_distance) + G2 * y_distance;
    B = B * (1 - y_distance) + B2 * y_distance;

    dst[coords]     = (unsigned char)(R + 0.5f);
    dst[coords + 1] = (unsigned char)(G + 0.5f);
    dst[coords + 2] = (unsigned char)(B + 0.5f);
}



// Applies linear blending between 2 images
extern "C" __global__
void linear_blend(
    unsigned char* dst,
    const unsigned char* src1,
    const unsigned char* src2,
    int width,
    int height,
    float distance
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int coords = (y * width + x) * 3;

    float R;
    float G;
    float B;

    // Calculate distance on one dimension
    R = src1[coords] * (1 - distance) + src2[coords] * distance;
    G = src1[coords + 1] * (1 - distance) + src2[coords + 1] * distance;
    B = src1[coords + 2] * (1 - distance) + src2[coords + 2] * distance;

    dst[coords] = (unsigned char)(R + 0.5f);
    dst[coords + 1] = (unsigned char)(G + 0.5f);
    dst[coords + 2] = (unsigned char)(B + 0.5f);
}