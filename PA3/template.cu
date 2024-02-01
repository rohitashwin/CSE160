#include <gputk.h>

#define ceil_div(x, y) (((x)-1) / (y) + 1)
#define BLOCKSIZE 32
#define gpuTKCheck(stmt)                                                \
  do                                                                    \
  {                                                                     \
    cudaError_t err = stmt;                                             \
    if (err != cudaSuccess)                                             \
    {                                                                   \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                    \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
      return -1;                                                        \
    }                                                                   \
  } while (0)

// General Matrix Multiplication - Resolve Bank
#define BLOCK_DIM_M 16
#define BLOCK_DIM_N 16
#define BLOCK_DIM_K 16
#define TILE_DIM_M 16
#define TILE_DIM_N 16

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  int K = numAColumns;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= numCRows || col >= numCColumns) return;
  float sum = 0;
  for (int i = 0; i < K; ++i)
  {
    sum += ((float**)(A)[row][i]) * ((float**)B[i][col]);
  }
  (float**)(C)[row][col] = sum;
}

int main(int argc, char **argv)
{
  gpuTKArg_t args;
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
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                               &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                               &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  gpuTKCheck(cudaMalloc((void **)&deviceA, sizeof(float) * numARows * numAColumns));
  gpuTKCheck(cudaMalloc((void **)&deviceB, sizeof(float) * numBRows * numBColumns));
  gpuTKCheck(cudaMalloc((void **)&deviceC, sizeof(float) * numCRows * numCColumns));
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  gpuTKCheck(cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice));
  gpuTKCheck(cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice));
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // dim3 dimGrid((numCColumns - 1) / 16 + 1, (numCRows - 1) / 16 + 1, 1);
  dim3 dimGrid(ceil_div(numCColumns, BLOCKSIZE), ceil_div(numCRows, BLOCKSIZE), 1);
  dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows,
                                        numAColumns, numBRows,
                                        numBColumns, numCRows,
                                        numCColumns);
  gpuTKCheck(cudaGetLastError());
  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  gpuTKCheck(cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost));
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}