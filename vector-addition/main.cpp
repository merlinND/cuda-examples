// MP 1
#include <wb.h>

/**
 * Code stub made available on webgpu.com
 * by the Heterogeneous Parallel Programming MOOC teaching team
 */

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  int inputSize = inputLength * sizeof(float);
  hostOutput = ( float * )malloc(inputSize);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  // Allocate GPU memory to:
  // - hold the first input (vector `A` of size `inputLength`)
  cudaMalloc((void **) &deviceInput1, inputSize);
  // - hold the second input (vector `B` of size `inputLength`)
  cudaMalloc((void **) &deviceInput2, inputSize);
  // - hold the result (vector `C` of size `inputLength`)
  cudaMalloc((void **) &deviceOutput, inputSize);
  wbTime_stop(GPU, "Allocating GPU memory.");


  wbTime_start(GPU, "Copying input memory to the GPU.");
  // Copy inputs `A` and `B` to GPU shared memory
  cudaMemcpy(deviceInput1, hostInput1, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputSize, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Initialize the grid and block dimensions
  int threadsPerBlock = 256;
  int nBlocks = (inputSize - 1) / threadsPerBlock + 1;
  wbTime_start(Compute, "Performing CUDA computation");
  // Launching GPU kernel (will spawn as many threads as needed)
  vecAdd<<<nBlocks, threadsPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  // Copy output back to host memory
  cudaMemcpy(hostOutput, deviceOutput, inputSize, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  // Free GPU memory
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
