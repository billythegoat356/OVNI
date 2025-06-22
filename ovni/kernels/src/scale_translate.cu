extern "C" __global__
void scale_translate(
    const unsigned char* src,
    int srcw,
    int srch,
    unsigned char* dst,
    int dstw,
    int dsth,
    float scale,
    int tx,
    int ty
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstw || y >= dsth) return;

    // Destination
    int flattened_coords = (y * dstw + x) * 3;

    // Calculate source coordinates from destination coordinates
    float sx = (x - tx) / scale;
    float sy = (y - ty) / scale;

    if (sx < 0 || sx + 1 > srcw || sy < 0 || sy + 1 > srch) {
        dst[flattened_coords] = (unsigned char)(0);
        dst[flattened_coords + 1] = (unsigned char)(0);
        dst[flattened_coords + 2] = (unsigned char)(0);
        return;
    }

    // Clamp to source image bounds
    sx = fmaxf(0.0f, fminf(sx, srcw - 1.0f));
    sy = fmaxf(0.0f, fminf(sy, srch - 1.0f));

    // Calculate surrounding bounds
    int left_x = int(floorf(sx));
    int right_x = min(left_x + 1, srcw - 1);

    int top_y = int(floorf(sy));
    int bottom_y = min(top_y + 1, srch - 1);

    // Calculate X coordinates averages
    float x_cond = sx - left_x;

    int flattened_top_left = (top_y * srcw + left_x) * 3;
    int flattened_top_right = (top_y * srcw + right_x) * 3;

    float R_top = src[flattened_top_left] * (1 - x_cond) + src[flattened_top_right] * x_cond;
    float G_top = src[flattened_top_left + 1] * (1 - x_cond) + src[flattened_top_right + 1] * x_cond;
    float B_top = src[flattened_top_left + 2] * (1 - x_cond) + src[flattened_top_right + 2] * x_cond;

    int flattened_bottom_left = (bottom_y * srcw + left_x) * 3;
    int flattened_bottom_right = (bottom_y * srcw + right_x) * 3;

    float R_bottom = src[flattened_bottom_left] * (1 - x_cond) + src[flattened_bottom_right] * x_cond;
    float G_bottom = src[flattened_bottom_left + 1] * (1 - x_cond) + src[flattened_bottom_right + 1] * x_cond;
    float B_bottom = src[flattened_bottom_left + 2] * (1 - x_cond) + src[flattened_bottom_right + 2] * x_cond;

    // Calculate Y coordinates average
    float y_cond = sy - top_y;

    float R = R_top * (1 - y_cond) + R_bottom * y_cond;
    float G = G_top * (1 - y_cond) + G_bottom * y_cond;
    float B = B_top * (1 - y_cond) + B_bottom * y_cond;

    // Update in destination
    dst[flattened_coords] = (unsigned char)(roundf(R));
    dst[flattened_coords + 1] = (unsigned char)(roundf(G));
    dst[flattened_coords + 2] = (unsigned char)(roundf(B));
    
}