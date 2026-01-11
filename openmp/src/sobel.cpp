#include "sobel.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#include "stb_image.h"
#include "stb_image_write.h"

#define SOBEL_MASK_DIM 3
#define SOBEL_MASK_SIZE (SOBEL_MASK_DIM * SOBEL_MASK_DIM)
#define CHANNELS STBI_rgb_alpha

const int sobelX[SOBEL_MASK_SIZE] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

const int sobelY[SOBEL_MASK_SIZE] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

static bool isOutOfBounds(int x, int y, int width, int height) {
  return (x < 0 || y < 0 || x >= width || y >= height);
}

static void getPixel(const stbi_uc *image, int width, int x, int y,
                     Pixel *pixel) {
  const stbi_uc *p = image + CHANNELS * (y * width + x);
  pixel->r = p[0];
  pixel->g = p[1];
  pixel->b = p[2];
  pixel->a = p[3];
}

static void setPixel(stbi_uc *image, int width, int x, int y,
                     const Pixel *pixel) {
  stbi_uc *p = image + CHANNELS * (y * width + x);
  p[0] = pixel->r;
  p[1] = pixel->g;
  p[2] = pixel->b;
  p[3] = pixel->a;
}

static void convolveAtPixel(const stbi_uc *input_img, stbi_uc *output_img,
                            int width, int height, const int *mask,
                            int mask_dimension, int x, int y) {
  int red = 0, green = 0, blue = 0;
  const int mask_size = mask_dimension / 2;

  for (int i = 0; i < mask_dimension; ++i) {
    for (int j = 0; j < mask_dimension; ++j) {
      int cx = x - mask_size + i;
      int cy = y - mask_size + j;

      if (isOutOfBounds(cx, cy, width, height))
        continue;

      int coeff = mask[i * mask_dimension + j];

      Pixel p;
      getPixel(input_img, width, cx, cy, &p);

      red += p.r * coeff;
      green += p.g * coeff;
      blue += p.b * coeff;
    }
  }

  Pixel out;
  out.r = static_cast<stbi_uc>(std::clamp(red, 0, 255));
  out.g = static_cast<stbi_uc>(std::clamp(green, 0, 255));
  out.b = static_cast<stbi_uc>(std::clamp(blue, 0, 255));
  out.a = 255;

  setPixel(output_img, width, x, y, &out);
}

static void combineAtPixel(const stbi_uc *gx_img, const stbi_uc *gy_img,
                           stbi_uc *output_img, int width, int x, int y) {
  Pixel gx, gy;
  getPixel(gx_img, width, x, y, &gx);
  getPixel(gy_img, width, x, y, &gy);

  int mag = static_cast<int>(
      std::sqrt(static_cast<float>(gx.r * gx.r + gy.r * gy.r)));

  mag = std::clamp(mag, 0, 255);

  Pixel out;
  out.r = out.g = out.b = static_cast<stbi_uc>(mag);
  out.a = 255;

  setPixel(output_img, width, x, y, &out);
}

void convolve(const stbi_uc *input_img, stbi_uc *output_img, int width,
              int height, const int *mask, int mask_dimension) {

#pragma omp parallel for schedule(static)
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      convolveAtPixel(input_img, output_img, width, height, mask,
                      mask_dimension, x, y);
}

void combineGradients(const stbi_uc *gx_img, const stbi_uc *gy_img,
                      stbi_uc *output_img, int width, int height) {
#pragma omp parallel for schedule(static)
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      combineAtPixel(gx_img, gy_img, output_img, width, x, y);
}

void edgeDetection(stbi_uc *input, stbi_uc *output, int width, int height) {
  const size_t img_size = static_cast<size_t>(width) * height * CHANNELS;

  auto input_buf = std::make_unique<stbi_uc[]>(img_size);
  auto grad_x = std::make_unique<stbi_uc[]>(img_size);
  auto grad_y = std::make_unique<stbi_uc[]>(img_size);
  auto output_buf = std::make_unique<stbi_uc[]>(img_size);

  std::memcpy(input_buf.get(), input, img_size);

  convolve(input_buf.get(), grad_x.get(), width, height, sobelX,
           SOBEL_MASK_DIM);
  convolve(input_buf.get(), grad_y.get(), width, height, sobelY,
           SOBEL_MASK_DIM);
  combineGradients(grad_x.get(), grad_y.get(), output_buf.get(), width, height);

  std::memcpy(output, output_buf.get(), img_size);
}
