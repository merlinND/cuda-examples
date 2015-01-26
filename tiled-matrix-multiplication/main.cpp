#include <wb.h>

/**
 * Code made available on webgpu.com
 * by the Heterogeneous Parallel Programming MOOC teaching team
 */

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  // Each thread computes a single element of the output matrix C
  // These shared sub-matrices hold a single tile at a time
  __shared__ float d_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float d_B[TILE_WIDTH][TILE_WIDTH];

  // Aliases
  int n = numAColumns; // == numBRows
  int tX = threadIdx.x,
      tY = threadIdx.y;

  // Indices of the output cell we're computing
  int row = (blockIdx.y * blockDim.y) + tY;
  int col = (blockIdx.x * blockDim.x) + tX;
  float accumulator = 0.f;

  // For each tile
  for(int t = 0; t < (n - 1) / TILE_WIDTH + 1; ++t) {
    // Collaboratively load the tile (defaulting to 0 is we're out of boundaries)
    // Note that we require thread blocks to be of size TILE_WIDTH * TILE_WIDTH

    // Offset of this tile's corner (common to the thread block):
    //   t * TILE_WIDTH
    // Indices of the particular elements to load in the tile (unique to this thread):
    int jA = (t * TILE_WIDTH) + tX,
        iB = (t * TILE_WIDTH) + tY;
    if(row < numARows && jA < numAColumns) {
      d_A[tY][tX] = A[row * numAColumns + jA];
    }
    else {
      d_A[tY][tX] = 0.f;
    }
    if(iB < numBRows && col < numBColumns) {
      d_B[tY][tX] = B[iB * numBColumns + col];
    }
    else {
      d_B[tY][tX] = 0.f;
    }

    // Wait for collaborative loading to end
    __syncthreads();

    // Compute this tile's dot-product segment using only shared memory accesses
    for(int k = 0; k < TILE_WIDTH; ++k) {
      accumulator += d_A[tY][k] * d_B[k][tX];
    }

    // Wait for other threads before going to the next tile
    __syncthreads();
  }

  // Output only if this cell is within bounds
  if(row < numCRows && col < numCColumns) {
    C[row * numCColumns + col] = accumulator;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  // C = A * B
  // Output matrix dimensions & dynamic allocation
  numCRows = numARows;
  numCColumns = numBColumns;
  hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  // Device memory allocation (A, B and C matrices)
  wbCheck(cudaMalloc((void **) &deviceA, sizeof(float) * numARows * numAColumns));
  wbCheck(cudaMalloc((void **) &deviceB, sizeof(float) * numBRows * numBColumns));
  wbCheck(cudaMalloc((void **) &deviceC, sizeof(float) * numCRows * numCColumns));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  // Transfer input matrices to the device
  wbCheck(cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Compute thread grid dimensions
  dim3 gridSize( (numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
  dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  // Launch the kernel
  matrixMultiplyShared<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC,
                                                numARows, numAColumns, numBRows,
                                                numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  // Retrieve output
  wbCheck(cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  // Free the device memory
  wbCheck(cudaFree(deviceA));
  wbCheck(cudaFree(deviceB));
  wbCheck(cudaFree(deviceC));
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
