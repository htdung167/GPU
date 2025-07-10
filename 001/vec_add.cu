

__global__ void vecAddKernel(float* A, float* B, float*C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Part 1: Allocate device memory for A, B, and C
    cudaError_t err = cudaMalloc((void**)&A_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    // Copy A and B to device memory
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Part 2: Call kernel – to launch a grid of threads
    // to perform the actual vector addition
    vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    // Part 3: Copy C from the device memory
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Free device vectors
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}