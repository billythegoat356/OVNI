

__device__
void bilinear(
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
    if (x < 0 || x + 1 > width || y < 0 || y + 1 > height) {
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