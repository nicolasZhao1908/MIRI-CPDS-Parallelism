#include <math.h>
#include <cuda.h>

__global__ void gpu_Heat(double *u, double *utmp, int N) {
    int i = (blockIdx.y * blockDim.y) + threadIdx.y;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i > 0 && i < N-1 && j > 0 && j < N-1){
        utmp[i*N+j]= 0.25 * (u[ i*N     + (j-1) ]+  // left
                u[ i*N     + (j+1) ]+  // right
                u[ (i-1)*N + j     ]+  // top
                u[ (i+1)*N + j     ]); // bottom
    }
}


__global__ void gpu_Diff(double *u, double *utmp, double* diffs, int N) {
    int i = (blockIdx.y * blockDim.y) + threadIdx.y;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i > 0 && i < N-1 && j > 0 && j < N-1){
        utmp[i*N+j]= 0.25 * (u[ i*N     + (j-1) ]+  // left
                u[ i*N     + (j+1) ]+  // right
                u[ (i-1)*N + j     ]+  // top
                u[ (i+1)*N + j     ]); // bottom
        diffs[(i-1)*(N-2)+j-1] = utmp[i*N+j] - u[i*N+j];
        diffs[(i-1)*(N-2)+j-1] *= diffs[(i-1)*(N-2)+j-1];
    }
}

__global__ void gpu_Heat_reduction(double *idata, double *odata, int N) {
	extern __shared__ double sdata[];
	unsigned int s;

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	unsigned int gridSize = blockDim.x * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < N) {
		sdata[tid] += idata[i] + idata[i + blockDim.x];
		i += gridSize;
	}
	__syncthreads();

	for (s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32) {
		volatile double *smem = sdata;

		smem[tid] += smem[tid + 32];
		smem[tid] += smem[tid + 16];
		smem[tid] += smem[tid + 8];
		smem[tid] += smem[tid + 4];
		smem[tid] += smem[tid + 2];
		smem[tid] += smem[tid + 1];
	}

	if (tid == 0)
		odata[blockIdx.x] = sdata[0];
}
