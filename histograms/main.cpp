// Histogram Equalization

#include    <wb.h>

/**
 * Uses code made available on webgpu.com
 * by the Heterogeneous Parallel Programming MOOC teaching team
 */

#define HISTOGRAM_LENGTH 256
#define TILE_SIZE 16 // TILE_SIZE ^ 2 must be >= HISTOGRAM_LENGTH
#define N_CHANNELS 3

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

typedef unsigned long long histogram_count;
typedef unsigned char uchar;

__device__ void getIndices(int * ti, int * tj, int * i, int * j) {
    // Various useful indices
    (*ti) = threadIdx.y;
    (*tj) = threadIdx.x;
    (*i ) = (blockIdx.y * TILE_SIZE + (*ti)) * N_CHANNELS;
    (*j ) = (blockIdx.x * TILE_SIZE + (*tj)) * N_CHANNELS;
}

__device__ void loadImageTile(float * inputImage, uchar * imageTile,
                              int width, int height,
                              int ti, int tj, int i, int j) {
    // Collaborative loading of this block's image tile
    // At the same time, we convert the image from [0; 1] values to [0; 255] values
    if(i < height && j < width) {
        for(int k = 0; k < N_CHANNELS; ++k) {
            float value = inputImage[(i * width + j) + k];
            imageTile[(ti * TILE_SIZE + tj) * N_CHANNELS + k] = (uchar)(value * 255);
        }
    }
    else {
        for(int k = 0; k < N_CHANNELS; ++k) {
            imageTile[(ti * TILE_SIZE + tj) * N_CHANNELS + k] = 0;
        }
    }
}

/**
 *
 * @param inputImage Interleaved RGB data, row-major order
 * @param histogram Histogram data (size HISTOGRAM_LENGTH)
 */
__global__ void computeGrayscaleHistogram(float * inputImage, histogram_count * histogram,
                                          int width, int height) {
    int ti, tj, i, j;
    getIndices(&ti, &tj, &i, &j);
    int localIndex = ti * TILE_SIZE + tj;

    // Collaborative loading to shared memory
    __shared__ uchar imageTile[TILE_SIZE][TILE_SIZE][N_CHANNELS];
    loadImageTile(inputImage, &(imageTile[0][0][0]), width, height, ti, tj, i, j);

    // Initializing the local copy histogram accumulator
    __shared__ histogram_count localHistogram[HISTOGRAM_LENGTH];
    if(localIndex < HISTOGRAM_LENGTH) {
        localHistogram[localIndex] = 0;
    }
    __syncthreads();

    // Accumulate histogram values (locally, in order to
    // reduce the number of atomic operations)
    // Simultaneously, we convert to luminosity value
    // Warning: we assume an RGB image
    uchar value = (uchar)(0.21f * imageTile[ti][tj][0]
                        + 0.71f * imageTile[ti][tj][1]
                        + 0.07f * imageTile[ti][tj][2]);
    atomicAdd(&(localHistogram[value]), 1);
    __syncthreads();

    // Output histogram values to global memory (must use slower atomic operations)
    if(localIndex < HISTOGRAM_LENGTH) {
        atomicAdd(&(histogram[localIndex]), localHistogram[localIndex]);
    }
    __syncthreads();
}

__global__ void cumulativeDistributionFunction(histogram_count * histogram, float * distribution,
                                               int width, int height) {
    int tx = threadIdx.x; // Goes up to (HISTOGRAM_LENGTH / 2) - 1
    // Collaborative loading, each thread handles two elements
    __shared__ float local[HISTOGRAM_LENGTH];
    local[tx * 2] = histogram[tx * 2] / (float)(width * height);
    local[tx * 2 + 1] = histogram[tx * 2 + 1] / (float)(width * height);
    __syncthreads();

    // ----- Prefix sum computation

    // 1. Reduction tree
    for(int stride = 1; stride < HISTOGRAM_LENGTH; stride *= 2) {
        int index = (tx + 1) * stride * 2 - 1;
        if(index < HISTOGRAM_LENGTH) {
            local[index] = local[index] + local[index - stride];
        }
        __syncthreads();
    }

    // 2. Fixup step
    for(int stride = HISTOGRAM_LENGTH / 2; stride >= 1; stride /= 2) {
        int index = (tx + 1) * stride * 2 - 1;

        // At each step, twice as many threads go to use
        if(index + stride < HISTOGRAM_LENGTH) {
            local[index + stride] = local[index + stride] + local[index];
        }
        __syncthreads();
    }

    // ----- Output
    distribution[tx * 2] = local[tx * 2];
    distribution[tx * 2 + 1] = local[tx * 2 + 1];
}

/**
 * Find the minimum nonzero value of the CDF
 */
__global__ void findDistributionMin(float * distribution, float * minValue) {
    int tx = threadIdx.x; // Goes up to (HISTOGRAM_LENGTH / 2) - 1
    // Collaborative loading, each thread handles two elements
    __shared__ float local[HISTOGRAM_LENGTH];
    local[tx * 2] = distribution[tx * 2];
    local[tx * 2 + 1] = distribution[tx * 2 + 1];
    __syncthreads();

    // ----- List reduction over `distribution` using the `min` operation
    for(int stride = 1; stride < HISTOGRAM_LENGTH; stride *= 2) {
        int index = (tx * stride * 2);
        if(index + stride < HISTOGRAM_LENGTH) {
            // We want the minimal nonzero value
            if(local[index] == 0) {
                local[index] = local[index + stride];
            }
            else {
                local[index] = min(local[index], local[index + stride]);
            }
        }
        __syncthreads();
    }

    // ----- Output
    if(tx == 0) {
        (*minValue) = local[0];
    }
    __syncthreads();
}

__global__ void histogramEqualization(float * inputImage, float * outputImage,
                                      float * distribution, float * distributionMin,
                                      int width, int height) {
    int ti, tj, i, j;
    getIndices(&ti, &tj, &i, &j);

    float base = (*distributionMin);

    // Color correction to obtain a linear cumulative distribution function
    if(i < height && j < width) {
        // TODO: i and j have already been multiplied by N_CHANNELS
        int index = (i * width * N_CHANNELS + j);
        for(int k = 0; k < N_CHANNELS; ++k) {
            uchar oldValue = (uchar)(inputImage[index + k] * 255);
            float newValue = (distribution[oldValue] - base) / (1.f - base);
            // Clamp value to the acceptable range
            newValue = fmin(fmax(newValue, 0.f), 1.f);

            // Output
            outputImage[index + k] = newValue;
        }
    }
    __syncthreads();
}

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
    float * deviceDistribution; // Cumulative distribution
    float * distributionMin; // Minimum nonzero value of the CDF
    float * deviceOutputImageData;
    const char * inputImageFile;
    int imageSize, histogramSize, distributionSize;

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
    distributionSize = HISTOGRAM_LENGTH * sizeof(float);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    // WARNING: we assume imageChannels == 3
    assert(imageChannels == N_CHANNELS);
    assert(TILE_SIZE * TILE_SIZE >= HISTOGRAM_LENGTH);

    wbLog(TRACE, "Image dimensions (width, height): ", imageWidth, ", ", imageHeight);


    wbTime_start(GPU, "Doing GPU memory allocation");
    wbCheck(cudaMalloc((void **) &deviceInputImageData, imageSize));
    wbCheck(cudaMalloc((void **) &deviceHistogram, histogramSize));
    wbCheck(cudaMalloc((void **) &deviceDistribution, distributionSize));
    wbCheck(cudaMalloc((void **) &distributionMin, sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceOutputImageData, imageSize));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize, cudaMemcpyHostToDevice));
    wbCheck(cudaMemset(deviceHistogram, 0, histogramSize));
    wbCheck(cudaMemset(deviceDistribution, 0, distributionSize));
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

    // Step 2: cumulative distribution function of the histogram
    // This is equivalent to a scan (i.e. prefix-sum) operation
    cumulativeDistributionFunction<<<1, HISTOGRAM_LENGTH / 2>>>(deviceHistogram, deviceDistribution,
                                                                imageWidth, imageHeight);
    cudaDeviceSynchronize();

    // Step 3: find the minimum nonzero value of the CDF
    // in order to be able to rescale it
    // This is equivalent to a list reduction using the `min` operation
    findDistributionMin<<<1, HISTOGRAM_LENGTH / 2>>>(deviceDistribution, distributionMin);
    cudaDeviceSynchronize();

    // Step 4: actual histogram equalization
    histogramEqualization<<<gridSize, blockSize>>>(deviceInputImageData, deviceOutputImageData,
                                                   deviceDistribution, distributionMin,
                                                   imageWidth, imageHeight);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data back from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
               imageSize,
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data back from the GPU");


    wbTime_start(GPU, "Freeing memory");
    wbCheck(cudaFree(deviceInputImageData));
    wbCheck(cudaFree(deviceHistogram));
    wbCheck(cudaFree(deviceDistribution));
    wbCheck(cudaFree(distributionMin));
    wbCheck(cudaFree(deviceOutputImageData));
    wbTime_stop(GPU, "Freeing memory");


    wbSolution(args, outputImage);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
