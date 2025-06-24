#include <stdint.h>

extern "C" __global__
void blend_ass_image(
    int w, int h,
    int stride,
    const unsigned char* bitmap,
    uint32_t color,
    int dst_x, int dst_y,
    int dst_w, int dst_h,
    unsigned char* dst
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    // Define bitmap & destination coords
    int b_coords = y * stride + x;
    int d_coords = (y + dst_y) * dst_w + (x + dst_x);

    // Get alpha value
    unsigned char alpha = bitmap[b_coords];

    // Return if its transparent
    if (alpha == 0) return;

    // Calculate color
    unsigned char R, G, B, A;

    R = color >> 24;
    G = (color >> 16) & 0xff;
    B = (color >> 8) & 0xff;
    A = (255 - (color & 0xff));

    // Alpha of the overlay, normalized to 0-1
    float ov_alpha = (float(alpha) * float(A)) / (255 * 255);

    // Original alpha
    float og_alpha = float(dst[4 * d_coords + 3]) / 255;

    // Calculate combined alpha
    float comb_alpha = ov_alpha + og_alpha * (1 - ov_alpha);

    // Precompute coefficients for color channels (for scaling back to full opacity)
    float coef_ov = ov_alpha / comb_alpha;
    float coef_og = og_alpha / comb_alpha;

    // Update alpha channel
    dst[4 * d_coords + 3] = (unsigned char)(comb_alpha * 255);

    // Update color channels
    dst[4 * d_coords] = (unsigned char)(R * coef_ov + dst[4 * d_coords] * coef_og);
    dst[4 * d_coords + 1] = (unsigned char)(G * coef_ov + dst[4 * d_coords + 1] * coef_og);
    dst[4 * d_coords + 2] = (unsigned char)(B * coef_ov + dst[4 * d_coords + 2] * coef_og);
}
