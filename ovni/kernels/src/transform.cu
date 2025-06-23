# include "interpolation.cu"

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

    unsigned char R;
    unsigned char G;
    unsigned char B;

    bilinear(src, srcw, srch, sx, sy, &R, &G, &B);

    // Update in destination
    dst[flattened_coords] = R;
    dst[flattened_coords + 1] = G;
    dst[flattened_coords + 2] = B;
}


extern "C" __global__
void resize(
    const unsigned char* src,
    int srcw, 
    int srch,
    unsigned char* dst,
    int dstw,
    int dsth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstw || y >= dsth) return;

    // Destination
    int flattened_coords = (y * dstw + x) * 3;

    // Calculate source coordinates
    float sx = (float(srcw) / dstw) * x;
    float sy = (float(srch) / dsth) * y;

    unsigned char R;
    unsigned char G;
    unsigned char B;

    bilinear(src, srcw, srch, sx, sy, &R, &G, &B);

    // Update in destination
    dst[flattened_coords] = R;
    dst[flattened_coords + 1] = G;
    dst[flattened_coords + 2] = B;
}


extern "C" __global__
void overlay_opacity(
    unsigned char* src,
    const unsigned char* overlay,
    int width,
    int height,
    float opacity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Destination
    int flattened_coords = (y * width + x) * 3;

    // Get average values with the opacity
    unsigned char R = overlay[flattened_coords] * opacity + src[flattened_coords] * (1 - opacity);
    unsigned char G = overlay[flattened_coords + 1] * opacity + src[flattened_coords + 1] * (1 - opacity);
    unsigned char B = overlay[flattened_coords + 2] * opacity + src[flattened_coords + 2] * (1 - opacity);

    // Update in source
    src[flattened_coords] = R;
    src[flattened_coords + 1] = G;
    src[flattened_coords + 2] = B;
}


extern "C" __global__
void blend(
    unsigned char* src,
    const unsigned char* overlay,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Destination
    int ov_flattened_coords = (y * width + x) * 4;
    int src_flattened_coords = (y * width + x) * 3;

    // Get average values with the opacity
    float opacity = float(overlay[ov_flattened_coords + 3]) / 255.0f;

    unsigned char R = overlay[ov_flattened_coords] * opacity + src[src_flattened_coords] * (1 - opacity);
    unsigned char G = overlay[ov_flattened_coords + 1] * opacity + src[src_flattened_coords + 1] * (1 - opacity);
    unsigned char B = overlay[ov_flattened_coords + 2] * opacity + src[src_flattened_coords + 2] * (1 - opacity);

    // Update in source
    src[src_flattened_coords] = R;
    src[src_flattened_coords + 1] = G;
    src[src_flattened_coords + 2] = B;
}