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