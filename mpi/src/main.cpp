#include "image.h"
#include "sobel.hpp"
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
  const char *inputPath = "lena.png";
  const char *outputPath = "lena_edges.png";

  int width, height, channels;

  MPI_Init(&argc, &argv);

  // Load image
  stbi_uc *inputImage = loadImage(inputPath, &width, &height, &channels);
  if (!inputImage) {
    fprintf(stderr, "Failed to load image %s\n", inputPath);
    return 1;
  }
  printf("Loaded image: %s (%dx%d, %d channels)\n", inputPath, width, height,
         channels);

  // Ensure 4 channels (RGBA)
  if (channels != 4) {
    printf("Warning: input image channels = %d. Edge detection will assume 4 "
           "channels (RGBA).\n",
           channels);
  }

  // Allocate output image
  stbi_uc *outputImage = new stbi_uc[width * height * 4]; // 4 channels (RGBA)

  // Run edge detection
  edgeDetection(inputImage, outputImage, width, height);

  // Save result
  writeImage(outputPath, outputImage, width, height, 4);
  printf("Saved edge-detected image as: %s\n", outputPath);

  // Free memory
  imageFree(inputImage);
  delete[] outputImage;

  MPI_Finalize();
}
