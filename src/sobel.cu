#include "sobel.h"
#include <math.h>
#include <cuda_runtime.h>

#define SOBEL_MASK_DIM 3
#define SOBEL_MASK_SIZE (SOBEL_MASK_DIM * SOBEL_MASK_DIM)
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16


const sobelX[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

const sobelY[9] = {
    1,  2,  1,
    0,  0,  0,
   -1, -2, -1
};

void checkCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess) {
        return;
    }
    fprintf(stderr, "%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err), err, file, line);
    exit(1);
}

#define CUDA_CHECK_RETURN(value) checkCudaErrorAux(__FILE__,__LINE__, #value, value)

/*
 * Kernel, that performs convolution
 */
__global__ void convolve(const stbi_uc* input_img, stbi_uc* output_img, const int width, const int height, 
                        const int* mask, const int mask_dimension){

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    int mask_size = mask_dimension / 2; //for mask 3x3 mask_size = 1

    Pixel current_pixel; output_pixel;
    int red = 0; int blue = 0; int green = 0; // int alpha = 0;

    int current_x; int current_y;
    int current_mask_elem;

    for(int i = 0; i < mask_dimension; i++){
        for(int j = 0; j < mask_dimension; j++){
            current_x = x - mask_size + i;
            current_y = y - mask_size + j;

            if(isOutOfBounds(current_x, current_y, width, height)){continue;}
            
            current_mask_elem = mask[i * mask_dimension + j];

            getPixel(input_img, width, current_x, current_y, &current_pixel);
            red += current_pixel.r * current_mask_elem;
            blue += current_pixel.b * current_mask_elem;
            green += current_pixel.g * current_mask_elem;
        }   
    }

    output_pixel.r = min(max(red, 0), 255);
    output_pixel.g = min(max(green, 0), 255);
    output_pixel.b = min(max(blue, 0), 255);
    setPixel(output_img, width, x, y, &output_pixel);
}

__global__ void combineGradients(const stbi_uc* gx_img, const stbi_uc* gy_img, stbi_uc* output_img, int width,
                                int height) { 

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) {return;} 

    Pixel gx_pixel, gy_pixel, out_pixel; 
    getPixel(gx_img, width, x, y, &gx_pixel);
    getPixel(gy_img, width, x, y, &gy_pixel);

    int mag = (int)sqrtf((float)(gx_pixel.r * gx_pixel.r + gy_pixel.r * gy_pixel.r));
    mag = min(max(mag, 0), 255); 
    out_pixel.r = out_pixel.g = out_pixel.b = mag; 
    setPixel(output_img, width, x, y, &out_pixel);
}

void edgeDetection(stbi_uc* input, stbi_uc* output, int width, int height) { 

    stbi_uc *d_input = nullptr, *d_grad_x = nullptr, *d_grad_y = nullptr, *d_output = nullptr;
    int *d_sobelX = nullptr, *d_sobelY = nullptr; 
    
    size_t img_size = (size_t)width * height * CHANNELS;

    // Allocate device memory
    CUDA_CHECK_RETURN(cudaMalloc(&d_input, img_size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_grad_x, img_size)); 
    CUDA_CHECK_RETURN(cudaMalloc(&d_grad_y, img_size)); 
    CUDA_CHECK_RETURN(cudaMalloc(&d_output, img_size)); 
    CUDA_CHECK_RETURN(cudaMalloc(&d_sobelX, SOBEL_MASK_SIZE * sizeof(int))); 
    CUDA_CHECK_RETURN(cudaMalloc(&d_sobelY, SOBEL_MASK_SIZE * sizeof(int))); 

    // Copy data 
    CUDA_CHECK_RETURN(cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_sobelX, sobelX, SOBEL_MASK_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_sobelY, sobelY, SOBEL_MASK_SIZE * sizeof(int), cudaMemcpyHostToDevice)); 
    
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    // Sobel X
    convolve<<<grid, block>>>(d_input, d_grad_x, width, height, d_sobelX, SOBEL_MASK_DIM); CUDA_CHECK_RETURN(cudaGetLastError());
     // Sobel Y
    convolve<<<grid, block>>>(d_input, d_grad_y, width, height, d_sobelY, SOBEL_MASK_DIM); CUDA_CHECK_RETURN(cudaGetLastError());

    // Combine 
    combineGradients<<<grid, block>>>(d_grad_x, d_grad_y, d_output, width, height); 
    CUDA_CHECK_RETURN(cudaGetLastError()); 

    // Copy back 
    CUDA_CHECK_RETURN(cudaMemcpy(output, d_output, img_size, cudaMemcpyDeviceToHost)); 

    // Free Memory
    cudaFree(d_input); 
    cudaFree(d_grad_x); 
    cudaFree(d_grad_y); 
    cudaFree(d_output); 
    cudaFree(d_sobelX);
    cudaFree(d_sobelY); 
}