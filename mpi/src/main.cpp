#include "image.h"
#include "sobel.hpp"
#include <mpi.h>
#include <stdio.h>
#include <memory>

int main(int argc, char **argv) {
  const char *inputPath = "lena.png";
  const char *outputPath = "lena_edges.png";

  int width, height, channels = 0;

  MPI_Init(&argc, &argv);

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  bool isRoot = (0 == myrank);

  std::unique_ptr<stbi_uc[]> input_buf;
  std::unique_ptr<stbi_uc[]> output_buf;

  // Load image
  if (isRoot) {
    input_buf = std::unique_ptr<stbi_uc[]>(loadImage(inputPath, &width, &height, &channels));
    if (!input_buf) {
      fprintf(stderr, "Failed to load image %s\n", inputPath);
      return 1;
    }
    printf("Loaded image: %s (%dx%d, %d channels)\n", inputPath, width, height,
          channels);

    if (channels != 4) {
      printf("Warning: input image channels = %d. Edge detection will assume 4 "
            "channels (RGBA).\n",
            channels);
    }

    output_buf = std::make_unique<stbi_uc[]>(width * height * 4); // 4 channels (RGBA)
  }
    
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

  edgeDetection(input_buf.get(), output_buf.get(), width, height);

  if (isRoot) {
    writeImage(outputPath, output_buf.get(), width, height, 4);
    printf("Saved edge-detected image as: %s\n", outputPath);
  }

  MPI_Finalize();
}
