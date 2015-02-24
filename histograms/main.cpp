// Histogram Equalization

#include    <wb.h>

/**
 * Uses code made available on webgpu.com
 * by the Heterogeneous Parallel Programming MOOC teaching team
 */

#define HISTOGRAM_LENGTH 256
#define TILE_SIZE 16

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

typedef unsigned long long histogram_count;

/**
 *
 * @param inputImage Interleaved RGB data, row-major order
 * @param histogram Histogram data (size HISTOGRAM_LENGTH)
 */
__global__ void computeGrayscaleHistogram(float * inputImage, histogram_count * histogram,
                                          int width, int height) {
    // TODO
}

__global__ void histogramEqualization(float * inputImage, float * outputImage, histogram_count * histogram,
                                      int width, int height) {
    // TODO
}

// TODO: use only one main kernel so as to need only one load to shared memory?

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * deviceInputImageData;
    histogram_count * deviceHistogram;
    float * deviceOutputImageData;
    const char * inputImageFile;
    int imageSize, histogramSize;

    //@@ Insert more code here

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    imageSize = imageWidth * imageHeight * imageChannels * sizeof(float);
    histogramSize = HISTOGRAM_LENGTH * sizeof(histogram_count);
    wbTime_stop(Generic, "Importing data and creating memory on host");


    wbTime_start(GPU, "Doing GPU memory allocation");
    wbCheck(cudaMalloc((void **) &deviceInputImageData, imageSize));
    wbCheck(cudaMalloc((void **) &deviceHistogram, histogramSize));
    wbCheck(cudaMalloc((void **) &deviceOutputImageData, imageSize));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize, cudaMemcpyHostToDevice));
    wbCheck(cudaMemset(deviceHistogram, 0, histogramSize));
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    // Compute tile size
    dim3 gridSize( (imageWidth - 1) / TILE_SIZE + 1,
                   (imageHeight - 1) / TILE_SIZE + 1, 1);
    dim3 blockSize( TILE_SIZE, TILE_SIZE, 1);

    // Step 1: compute the image's histogram
    // Simplification: we use the grayscale histogram
    computeGrayscaleHistogram<<<gridSize, blockSize>>>(deviceInputImageData,
                                                       deviceHistogram,
                                                       imageWidth, imageHeight);
    cudaDeviceSynchronize();

    // Step 2: histogram equalization
    histogramEqualization<<<gridSize, blockSize>>>(deviceInputImageData, deviceOutputImageData,
                                                   deviceHistogram,
                                                   imageWidth, imageHeight);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data back from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
               imageSize,
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data back from the GPU");

    wbSolution(args, outputImage);

    wbTime_start(GPU, "Freeing memory");
    wbCheck(cudaFree(deviceInputImageData));
    wbCheck(cudaFree(deviceHistogram));
    wbCheck(cudaFree(deviceOutputImageData));
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);
    wbTime_stop(GPU, "Freeing memory");

    return 0;
}
