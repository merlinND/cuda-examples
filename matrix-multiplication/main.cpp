#include <wb.h>

/**
 * Uses code made available on webgpu.com
 * by the Heterogeneous Parallel Programming MOOC teaching team
 */

/** Width of processes tiles in the GPU grid (in number of processes) **/
#define TILE_SIZE 16

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < numCRows && column < numCColumns) {
    float acc = 0;
    for(int i = 0; i < numAColumns; ++i) {
      acc += A[row * numAColumns + i] * B[i * numBColumns + column];
    }
    C[row * numCColumns + column] = acc;
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
  // Output matrix dimensions & allocation
  numCRows = numARows;
  numCColumns = numBColumns;
  hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  // Allocating GPU memory
  wbCheck(cudaMalloc((void **) &deviceA, sizeof(float) * numARows * numAColumns));
  wbCheck(cudaMalloc((void **) &deviceB, sizeof(float) * numBRows * numBColumns));
  wbCheck(cudaMalloc((void **) &deviceC, sizeof(float) * numCRows * numCColumns));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  // Copy input matrices to the GPU memory
  wbCheck(cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Compute grid and block dimension:
  // One process takes care of one of the output matrix's cell
  // We allocate a grid of many squares of size `TILE_SIZE`
  int gridWidth = (numCColumns - 1) / TILE_SIZE + 1;
  int gridHeight = (numCRows - 1) / TILE_SIZE + 1;
  dim3 gridSize(gridWidth, gridHeight, 1);
  dim3 tileSize(TILE_SIZE, TILE_SIZE, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  // Launch the GPU Kernel
  matrixMultiply<<<gridSize, tileSize>>>(deviceA, deviceB, deviceC, numARows, numAColumns,
                                         numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  // Retrieve the result from GPU memory
  wbCheck(cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  // Free GPU memory
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
