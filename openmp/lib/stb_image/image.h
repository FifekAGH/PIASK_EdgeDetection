#ifndef IMAGE_H
#define IMAGE_H

typedef struct Pixel {
    stbi_uc r;
    stbi_uc g;
    stbi_uc b;
    stbi_uc a;
} Pixel;

stbi_uc* loadImage(const char* path_to_image, int* width, int* height, int* channels);

void writeImage(const char* path_to_image, stbi_uc* image, int width, int height, int channels);

void imageFree(stbi_uc* image);

__host__ __device__ void getPixel(stbi_uc* image, int width, int x, int y, Pixel* pixel);

__host__ __device__ void setPixel(stbi_uc* image, int width, int x, int y, Pixel* pixel);

__device__ bool isOutOfBounds(int x, int y, int image_width, int image_height);

#endif