#include <math.h>
#include <cuda.h>

__global__ void gpu_Heat(double *u, double *utmp, int N) {
    int sizey = N;
    int i = (blockIdx.y * blockDim.y) + threadIdx.y;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i > 0 && i < N-1 && j > 0 && j < N-1){
        utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
                u[ i*sizey     + (j+1) ]+  // right
                u[ (i-1)*sizey + j     ]+  // top
                u[ (i+1)*sizey + j     ]); // bottom
    }
}
