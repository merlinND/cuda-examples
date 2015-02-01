#include    <wb.h>

/**
 * Code made available on webgpu.com
 * by the Heterogeneous Parallel Programming MOOC teaching team
 */

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

/**
 * Mask has size fixed to 5x5 in this MP to simplify
 */
#define MASK_WIDTH  5
#define MASK_RADIUS MASK_WIDTH/2

/**
 * Recall each GPU tile is larger than the number of output cells since we need
 * one thread to load each element used in the convolution. Some threads will not
 * output anything.
 */
#define OUTPUT_TILE_SIZE 12
#define TILE_SIZE 16 // == OUTPUT_TILE_SIZE + (MASK_WIDTH - 1)

/**
 * Image convolution kernel
 * Assume image data uses interleaved channels
 * Mask data is common to all channels
 */
__global__ void imageConvolution(float * inputImageData,
                                 int imageWidth, int imageHeight, int nChannels,
                                 float * outputImageData,
                                 float const * __restrict__ maskData) {
    // Local copy of the input image tile we're working onto
    __shared__ float localImage[TILE_SIZE][TILE_SIZE][nChannels];

    int bi = blockIdx.y,
        bj = blockIdx.x;
    int ti = threadIdx.y,
        tj = threadIdx.x;

    // Coordinates of the cell to load (input image)
    int inputI = bi * blockDim.y + ti - MASK_RADIUS,
        inputJ = bj * blockDim.x + tj - MASK_RADIUS;

    // Collaboratively load mask elements into shared memory
    // Boundary condition: replace non-existing elements by 0
    if(inputI >= 0 && inputI < imageHeight && inputJ >= 0 && inputJ < imageWidth) {
        // Load all channels
        int linearized = (inputI * imageWidth + inputJ) * nChannels;
        for(int k = 0; k < nChannels; ++k) {
            localImage[ti][tj][k] = inputImageData[linearized + k];
        }
    } else {
        for(int k = 0; k < nChannels; ++k) {
            localImage[ti][tj][k] = 0;
        }
    }
    __syncthreads();

    // Convolution: not all threads have an output to write
    if(ti < OUTPUT_TILE_SIZE && tj < OUTPUT_TILE_SIZE) {
        // Coordinates of the cell to output (output image)
        int outputI = bi * OUTPUT_TILE_SIZE + ti,
            outputJ = bj * OUTPUT_TILE_SIZE + tj;
        if(outputI < imageHeight && outputJ < imageWidth) {
            int linearized = (outputI * imageWidth + outputJ) * nChannels;

            // Coordinates of the top left convolution corner
            // in the local image tile for this output cell
            int cornerI = ti,
                cornerJ = tj;

            // Perform convolution on each channel using coefficients from `maskData`
            for(int k = 0; k < nChannels; ++k) {
                float accumulator = 0;
                for(int i = 0; i < MASK_WIDTH; ++i) {
                    for(int j = 0; j < MASK_WIDTH; ++j) {
                        accumulator += maskData[i][j] * localImage[cornerI + i][cornerJ + j][k];
                    }
                }

                outputImageData[linearized + k] = accumulator;
            }
        }
    }
    __syncthreads();
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == MASK_WIDTH);
    assert(maskColumns == MASK_WIDTH);
    assert(TILE_SIZE == OUTPUT_TILE_SIZE + (MASK_WIDTH - 1));

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    // Compute tile size
    dim3 gridSize( (imageHeight - 1) / OUTPUT_TILE_SIZE + 1,
                   (imageWidth - 1) / OUTPUT_TILE_SIZE + 1, 1);
    dim3 blockSize( TILE_SIZE, TILE_SIZE, 1);

    // Launch kernel
    imageConvolution<<<gridSize, blockSize>>>(deviceInputImageData,
                                              imageWidth, imageHeight, imageChannels,
                                              deviceOutputImageData,
                                              deviceMaskData);
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
