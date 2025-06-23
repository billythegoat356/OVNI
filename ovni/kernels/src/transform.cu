
extern "C" __global__
void scale_translate(
    const unsigned char* src,
    int srcw,
    int srch,
    unsigned char* dst,
    int dstw,
    int dsth,
    float scale,
    float tx,
    float ty
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

    bilinear_one(src, srcw, srch, sx, sy, &R, &G, &B);

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

    bilinear_one(src, srcw, srch, sx, sy, &R, &G, &B);

    // Update in destination
    dst[flattened_coords] = R;
    dst[flattened_coords + 1] = G;
    dst[flattened_coords + 2] = B;
}


__device__
void mix_channels(
    const unsigned char* arr1,
    const unsigned char* arr2,
    int coords1,
    int coords2,
    float opacity, // Opacity of the first array!
    unsigned char *R,
    unsigned char *G,
    unsigned char *B
) {
    *R = arr1[coords1] * opacity + arr2[coords2] * (1 - opacity);
    *G = arr1[coords1 + 1] * opacity + arr2[coords2 + 1] * (1 - opacity);
    *B = arr1[coords1 + 2] * opacity + arr2[coords2 + 2] * (1 - opacity);
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

    unsigned char R;
    unsigned char G;
    unsigned char B;

    mix_channels(overlay, src, flattened_coords, flattened_coords, opacity, &R, &G, &B);

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

    unsigned char R;
    unsigned char G;
    unsigned char B;

    mix_channels(overlay, src, ov_flattened_coords, src_flattened_coords, opacity, &R, &G, &B);

    // Update in source
    src[src_flattened_coords] = R;
    src[src_flattened_coords + 1] = G;
    src[src_flattened_coords + 2] = B;
}


