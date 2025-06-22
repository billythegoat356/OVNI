extern "C" __global__
void nv12_to_rgb(
    const unsigned char* y_plane,
    const unsigned char* uv_plane,
    unsigned char* rgb,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int y_idx = y * width + x;
    int uv_idx = (y / 2) * width + (x / 2) * 2;

    float Y = (float)y_plane[y_idx];
    float U = (float)uv_plane[uv_idx]     - 128.0f;
    float V = (float)uv_plane[uv_idx + 1] - 128.0f;

    // BT.709 full range (scale coefficients)
    float R = Y + 1.5748f * V;
    float G = Y - 0.1873f * U - 0.4681f * V;
    float B = Y + 1.8556f * U;

    R = min(max(R, 0.0f), 255.0f);
    G = min(max(G, 0.0f), 255.0f);
    B = min(max(B, 0.0f), 255.0f);

    int rgb_idx = (y * width + x) * 3;
    rgb[rgb_idx + 0] = (unsigned char)R;
    rgb[rgb_idx + 1] = (unsigned char)G;
    rgb[rgb_idx + 2] = (unsigned char)B;
}

extern "C" __global__
void rgb_to_nv12(
    const unsigned char* rgb,
    unsigned char* y_plane,
    unsigned char* uv_plane,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float R = (float)rgb[idx + 0];
    float G = (float)rgb[idx + 1];
    float B = (float)rgb[idx + 2];

    // BT.709 full range
    float Y = 0.2126f * R + 0.7152f * G + 0.0722f * B;
    float U = (B - Y) / 1.8556f + 128.0f;
    float V = (R - Y) / 1.5748f + 128.0f;

    Y = min(max(Y, 0.0f), 255.0f);
    U = min(max(U, 0.0f), 255.0f);
    V = min(max(V, 0.0f), 255.0f);

    y_plane[y * width + x] = (unsigned char)(Y);

    if ((x % 2 == 0) && (y % 2 == 0)) {
        int uv_idx = (y / 2) * width + x;
        uv_plane[uv_idx]     = (unsigned char)(U);
        uv_plane[uv_idx + 1] = (unsigned char)(V);
    }
}