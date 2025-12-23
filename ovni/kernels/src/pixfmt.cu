extern "C" __global__
void nv12_to_rgb(
    const unsigned char* y_plane,
    const unsigned char* uv_plane,
    unsigned char* rgb,
    int width,
    int height
) {
    // Y is expected to be within [16, 235] and U/V within [16, 240]

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int y_idx  = y * width + x;
    int uv_idx = (y / 2) * width + (x / 2) * 2;

    // LIMITED → FULL
    float Y = ((float)y_plane[y_idx] - 16.0f) * (255.0f / 219.0f);
    float U = ((float)uv_plane[uv_idx]     - 128.0f) * (255.0f / 224.0f);
    float V = ((float)uv_plane[uv_idx + 1] - 128.0f) * (255.0f / 224.0f);

    float R = Y + 1.5748f * V;
    float G = Y - 0.1873f * U - 0.4681f * V;
    float B = Y + 1.8556f * U;

    R = fminf(fmaxf(R, 0.0f), 255.0f);
    G = fminf(fmaxf(G, 0.0f), 255.0f);
    B = fminf(fmaxf(B, 0.0f), 255.0f);

    int rgb_idx = (y * width + x) * 3;
    rgb[rgb_idx + 0] = (unsigned char)(R + 0.5f);
    rgb[rgb_idx + 1] = (unsigned char)(G + 0.5f);
    rgb[rgb_idx + 2] = (unsigned char)(B + 0.5f);
}



extern "C" __global__
void rgb_to_nv12(
    const unsigned char* rgb,
    unsigned char* y_plane,
    unsigned char* uv_plane,
    int width,
    int height
) {
    // Y is expected to be within [16, 235] and U/V within [16, 240]
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float R = (float)rgb[idx + 0];
    float G = (float)rgb[idx + 1];
    float B = (float)rgb[idx + 2];

    // Luma full
    float Yf = 0.2126f * R + 0.7152f * G + 0.0722f * B;

    // FULL → LIMITED
    float Y = 16.0f + Yf * (219.0f / 255.0f);
    float U = 128.0f + ((B - Yf) / 1.8556f) * (224.0f / 255.0f);
    float V = 128.0f + ((R - Yf) / 1.5748f) * (224.0f / 255.0f);

    Y = fminf(fmaxf(Y, 16.0f), 235.0f);
    U = fminf(fmaxf(U, 16.0f), 240.0f);
    V = fminf(fmaxf(V, 16.0f), 240.0f);

    y_plane[y * width + x] = (unsigned char)(Y + 0.5f);

    if ((x % 2 == 0) && (y % 2 == 0)) {
        int uv_idx = (y / 2) * width + x;
        uv_plane[uv_idx]     = (unsigned char)(U + 0.5f);
        uv_plane[uv_idx + 1] = (unsigned char)(V + 0.5f);
    }
}
