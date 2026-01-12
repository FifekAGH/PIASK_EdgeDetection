#include "sobel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mpi.h>
#include <vector>

#include "stb_image.h"
#include "stb_image_write.h"

#define SOBEL_MASK_DIM 3
#define SOBEL_MASK_SIZE (SOBEL_MASK_DIM * SOBEL_MASK_DIM)
#define CHANNELS STBI_rgb_alpha

const int sobelX[SOBEL_MASK_SIZE] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1,
};

const int sobelY[SOBEL_MASK_SIZE] = {
     1,  2,  1,
     0,  0,  0,
    -1, -2, -1,
};

static bool isOutOfBounds(int x, int y, int width, int height) {
  return (x < 0 || y < 0 || x >= width || y >= height);
}

static void getPixel(const stbi_uc* image, int width, int x, int y,
                     Pixel* pixel) {
  const stbi_uc* p = image + CHANNELS * (y * width + x);
  pixel->r = p[0];
  pixel->g = p[1];
  pixel->b = p[2];
  pixel->a = p[3];
}

static void setPixel(stbi_uc* image, int width, int x, int y,
                     const Pixel* pixel) {
  stbi_uc* p = image + CHANNELS * (y * width + x);
  p[0] = pixel->r;
  p[1] = pixel->g;
  p[2] = pixel->b;
  p[3] = pixel->a;
}

static void convolveAtPixel(const stbi_uc* input_img,
                            stbi_uc* output_img,
                            int width,
                            int height,
                            const int* mask,
                            int mask_dimension,
                            int x,
                            int y) {
  int red = 0, green = 0, blue = 0;
  int radius = mask_dimension / 2;

  for (int j = 0; j < mask_dimension; ++j) {
    for (int i = 0; i < mask_dimension; ++i) {
      int cx = x + i - radius;
      int cy = y + j - radius;

      if (isOutOfBounds(cx, cy, width, height))
        continue;

      int coeff = mask[j * mask_dimension + i];

      Pixel p;
      getPixel(input_img, width, cx, cy, &p);

      red   += p.r * coeff;
      green += p.g * coeff;
      blue  += p.b * coeff;
    }
  }

  Pixel out;
  out.r = static_cast<stbi_uc>(std::clamp(red,   0, 255));
  out.g = static_cast<stbi_uc>(std::clamp(green, 0, 255));
  out.b = static_cast<stbi_uc>(std::clamp(blue,  0, 255));
  out.a = 255;

  setPixel(output_img, width, x, y, &out);
}

static void combineAtPixel(const stbi_uc* gx_img,
                           const stbi_uc* gy_img,
                           stbi_uc* output_img,
                           int width,
                           int x,
                           int y) {
  Pixel gx, gy;
  getPixel(gx_img, width, x, y, &gx);
  getPixel(gy_img, width, x, y, &gy);

  int gxv = static_cast<int>(gx.r);
  int gyv = static_cast<int>(gy.r);

  int mag = static_cast<int>(std::sqrt(gxv * gxv + gyv * gyv));
  mag = std::clamp(mag, 0, 255);

  Pixel out;
  out.r = out.g = out.b = static_cast<stbi_uc>(mag);
  out.a = 255;

  setPixel(output_img, width, x, y, &out);
}

void edgeDetection(stbi_uc* input,
                   stbi_uc* output,
                   int width,
                   int height) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (height % size != 0) {
    if (rank == 0) {
      fprintf(stderr,
              "Image height must be divisible by number of MPI processes\n");
    }
  }

  const int halo = 1;
  const int sub_height = height / size;
  const size_t row_bytes = width * CHANNELS;

  // Prepare scatter/gather parameters including halos
  std::vector<int> sendcounts, displs;
  if (rank == 0) {
    sendcounts.resize(size);
    displs.resize(size);

    // For each process, calculate number of rows with halos
    for (int r = 0; r < size; ++r) {
      int start = r * sub_height;
      int rows  = sub_height;

      if (r > 0)        rows += halo;
      if (r < (size - 1)) rows += halo;

      sendcounts[r] = rows * row_bytes;
      displs[r]     = (start - (r > 0 ? halo : 0)) * row_bytes;
    }

    printf("Running edge detection with %d MPI processes...\n", size);
  }

  int local_rows = sub_height + (rank > 0) + (rank < size - 1);
  size_t local_bytes = local_rows * row_bytes;

  auto local_buf = std::make_unique<stbi_uc[]>(local_bytes);

  // variable-length scatter with halos
  // sendcounts integer array (of length group size) specifying the number of elements to send to each processor
  // displs integer array (of length group size). Entry i specifies the displacement (relative to sendbuf from which to take the outgoing data to process i
  MPI_Scatterv(input,
               sendcounts.data(),
               displs.data(),
               MPI_UNSIGNED_CHAR,
               local_buf.get(),
               local_bytes,
               MPI_UNSIGNED_CHAR,
               0,
               MPI_COMM_WORLD);

  auto grad_x   = std::make_unique<stbi_uc[]>(local_bytes);
  auto grad_y   = std::make_unique<stbi_uc[]>(local_bytes);
  auto out_local = std::make_unique<stbi_uc[]>(local_bytes);

  const int y_start = (rank > 0) ? 1 : 0;
  const int y_end   = y_start + sub_height;

  for (int y = y_start; y < y_end; ++y) {
    for (int x = 0; x < width; ++x) {
      convolveAtPixel(local_buf.get(), grad_x.get(),
                      width, local_rows,
                      sobelX, SOBEL_MASK_DIM, x, y);
      convolveAtPixel(local_buf.get(), grad_y.get(),
                      width, local_rows,
                      sobelY, SOBEL_MASK_DIM, x, y);
    }
  }

  for (int y = y_start; y < y_end; ++y) {
    for (int x = 0; x < width; ++x) {
      combineAtPixel(grad_x.get(), grad_y.get(),
                     out_local.get(), width, x, y);

    }
  }

  // Variable-length gather of results without halos
  MPI_Gatherv(out_local.get() + y_start * row_bytes,
              sub_height * row_bytes,
              MPI_UNSIGNED_CHAR,
              output,
              sendcounts.data(),
              displs.data(),
              MPI_UNSIGNED_CHAR,
              0,
              MPI_COMM_WORLD);
}
