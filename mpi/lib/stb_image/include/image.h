#ifndef IMAGE_H
#define IMAGE_H

#include "stb_image.h"
#include "stb_image_write.h"

typedef struct Pixel {
    stbi_uc r;
    stbi_uc g;
    stbi_uc b;
    stbi_uc a;
} Pixel;

stbi_uc* loadImage(const char* path_to_image, int* width, int* height, int* channels);

void writeImage(const char* path_to_image, stbi_uc* image, int width, int height, int channels);

void imageFree(stbi_uc* image);

// __host__ __device__ void getPixel(const stbi_uc* image, int width, int x, int y, Pixel* pixel);

// __host__ __device__ void setPixel(stbi_uc* image, int width, int x, int y, Pixel* pixel);


#endif