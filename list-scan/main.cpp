#include    <wb.h>

/**
 * Uses code made available on webgpu.com
 * by the Heterogeneous Parallel Programming MOOC teaching team
 */

// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__device__ void listReduction(float * local) {
    // For each level of the reduction tree
    for(int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;

        // At each step, half the threads go unused. Note we keep alive
        // adjacent threads so as to minimize control divergence in warps.
        if(index < BLOCK_SIZE * 2) {
            local[index] = local[index] + local[index - stride];
        }
        __syncthreads();
    }
}

__device__ void postProcessing(float * local) {
    for(int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;

        // At each step, twice as many threads go to use
        if(index + stride < BLOCK_SIZE * 2) {
            local[index + stride] = local[index + stride] + local[index];
        }
        __syncthreads();
    }
}

__device__ void collaborativeLoad(float * destination, float * source, int len) {
    // Collaborative loading of the input vector (two element per thread)
    unsigned int tx = threadIdx.x,
                 index = (blockIdx.x * BLOCK_SIZE + tx) * 2;
    destination[tx * 2] = (index < len ? source[index] : 0);
    destination[tx * 2 + 1] = (index + 1 < len ? source[index + 1] : 0);
    __syncthreads();
}

__global__ void scan(float * input, float * offsets, int len) {
    unsigned int tx = threadIdx.x,
                 index = (blockIdx.x * BLOCK_SIZE + tx) * 2;

    __shared__ float local[BLOCK_SIZE * 2];
    collaborativeLoad(&(local[0]), input, len);

    // Phase 1: reduction tree (similar to a simple list reduction)
    listReduction(local);

    // Phase 2: post-processing
    // After execution, only some entries of `local` contain the correct
    // partial sum. We need to complete the missing ones.
    postProcessing(local);

    // Output last element into the `offsets` array for this block
    if(len > BLOCK_SIZE * 2) {
        if(tx == 0) {
            offsets[blockIdx.x] = local[BLOCK_SIZE * 2 - 1];
        }
    }

    // Save partial result
    input[index] = local[tx * 2];
    input[index + 1] = local[tx * 2 + 1];
}

__global__ void applyOffsets(float * incomplete, float * output, float * offsets, int len) {
    unsigned int tx = threadIdx.x,
                 index = (blockIdx.x * BLOCK_SIZE + tx) * 2;

    __shared__ float local[BLOCK_SIZE * 2];
    collaborativeLoad(&(local[0]), incomplete, len);

    __shared__ int offset;
    if(tx == 0) {
        offset = (blockIdx.x == 0 ? 0 : offsets[blockIdx.x - 1]);
    }
    __syncthreads();


    // Output the final result
    if(index < len) {
        output[index] = local[tx * 2] + offset;
    }
    if(index + 1 < len) {
        output[index + 1] = local[tx * 2 + 1] + offset;
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOffsets;
    float * deviceOutput;
    int numElements; // number of elements in the list
    int numBlocks; // number of input segments that we will handle

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    numBlocks = (numElements - 1) / BLOCK_SIZE + 1;
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOffsets, numBlocks*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    // Set all output elements to zero (default value)
    wbCheck(cudaMemset(deviceOffsets, 0, numBlocks*sizeof(float)));
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 gridSize(numBlocks, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    wbTime_start(Compute, "Performing CUDA computation");
    // Initial scan, performed on sections of the input separately
    scan<<<gridSize, blockSize>>>(deviceInput, deviceOffsets, numElements);
    cudaDeviceSynchronize();

    // Applying prefix-sum to the offsets
    // WARNING: this is not truly generic, we assume `numBlocks <= BLOCK_SIZE`
    wbAssert(numBlocks <= BLOCK_SIZE);
    scan<<<1, BLOCK_SIZE>>>(deviceOffsets, NULL, numBlocks);
    cudaDeviceSynchronize();

    // Applying the offsets to each section
    applyOffsets<<<gridSize, blockSize>>>(deviceInput, deviceOutput, deviceOffsets, numElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
