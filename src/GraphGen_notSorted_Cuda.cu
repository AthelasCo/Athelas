#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "GraphGen_notSorted_Cuda.h"

struct GlobalConstants {

    unsigned long long cudaDeviceNumEdges, cudaDeviceNumVertices;
    double* cudaDeviceProbs;
    unsigned* cudaDeviceOutput;
};

__device__ inline int updiv(int n, int d) {
    return (n+d-1)/d;
}

__constant__ GlobalConstants cuConstGraphParams;


__global__ void KernelGenerateEdges(const bool directedGraph,
        const bool allowEdgeToSelf, const bool sorted) {
    // std::uniform_int_distribution<>& dis, std::mt19937_64& gen,
    // std::vector<unsigned long long>& duplicate_indices
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    int blockYmin = (blockIdx.y * blockDim.y) ;
    int blockYmax = (blockIdx.y * blockDim.y) + blockDim.y;
    int blockXmin = (blockIdx.x * blockDim.x) ;
    int blockXmax = (blockIdx.x * blockDim.x) + blockDim.x;

    int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

    // short imageWidth = cuConstRendererParams.imageWidth;
    // short imageHeight = cuConstRendererParams.imageHeight;
    // int numCircles = cuConstRendererParams.numCircles;

    // float invWidth = 1.f / imageWidth;
    // float invHeight = 1.f / imageHeight;

    //  __shared__ uint shared_no_of_circles[THREADS_PER_BLOCK];
    //  __shared__ uint shared_output[THREADS_PER_BLOCK];
    // volatile __shared__ uint shared_scratch[2 * THREADS_PER_BLOCK];
    // volatile __shared__ uint shared_circle_index[THREADS_PER_BLOCK];
    // __shared__ float3 position[THREADS_PER_BLOCK];
    // __shared__ float radii[THREADS_PER_BLOCK];
    // __shared__ float3 colors[THREADS_PER_BLOCK];
    // int circlesPerThread = updiv(numCircles,THREADS_PER_BLOCK);
    // float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
    //                                                  invHeight * (static_cast<float>(pixelY) + 0.5f));
    // float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    // float4 existingColor = *imgPtr;
    // for (int i=0; i < circlesPerThread; i++) {
    //     int cIdx = i * THREADS_PER_BLOCK + threadIndex;
    //     shared_no_of_circles[threadIndex] = 0;

    //     if (cIdx < numCircles) {
    //         int cIdx3 = 3 * cIdx;
    //         float3 p = *(float3*)(&cuConstRendererParams.position[cIdx3]);
    //         float  rad = cuConstRendererParams.radius[cIdx];
    //         short minX = static_cast<short>(imageWidth * (p.x - rad));
    //         short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    //         short minY = static_cast<short>(imageHeight * (p.y - rad));
    //         short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;


    //             if(!(blockXmin > maxX || blockXmax < minX
    //             || blockYmin > maxY || blockYmax < minY)){
    //                 shared_no_of_circles[threadIndex]=1;
    //                 radii[threadIndex] = rad;
    //                 position[threadIndex] = p;
    //                 colors[threadIndex] = *(float3*)(&cuConstRendererParams.color[cIdx3]);
    //             }
            
    //     } 

    //     __syncthreads();

    //     sharedMemExclusiveScan(threadIndex, shared_no_of_circles, shared_output,
    //                           shared_scratch, THREADS_PER_BLOCK);

    //     __syncthreads();

    //     int numOverBlkCircles = shared_output[THREADS_PER_BLOCK - 1];
    //     if ( shared_no_of_circles[THREADS_PER_BLOCK - 1] == 1 )
    //         numOverBlkCircles += 1;

    //     if ( shared_no_of_circles[threadIndex] == 1 ) {
    //         shared_circle_index[shared_output[threadIndex]] = threadIndex;
    //     }

    //     __syncthreads();
        
    //     for (int j=0; j < numOverBlkCircles; j++) {
    //         int index = i * THREADS_PER_BLOCK + shared_circle_index[j];
    //             float3 p = position[shared_circle_index[j]];
    //             float rad = radii[shared_circle_index[j]];
    //             float3 color = colors[shared_circle_index[j]];
    //             shadePixel(pixelCenterNorm, p, &existingColor,rad, color);
    //     }


    // }
    // *imgPtr = existingColor;


}

////////////////////////////////////////////////////////////////////////////////////////



static inline int updivHost(int n, int d) {
    return (n+d-1)/d;
}


bool setup(
        const unsigned long long nEdges,
        const unsigned long long nVertices,
        const double RMAT_a, const double RMAT_b, const double RMAT_c,
        const unsigned int nCPUWorkerThreads,
        std::ofstream& outFile,
        const unsigned long long standardCapacity,
        const bool allowEdgeToSelf,
        const bool allowDuplicateEdges,
        const bool directedGraph,
        const bool sorted){
    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce GTX 480") == 0
            || name.compare("GeForce GTX 670") == 0
            || name.compare("GeForce GTX 780") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA GTX 480, 670 or 780.\n");
        printf("---------------------------------------------------------\n");
    }
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy
    double* cudaDeviceProbs = NULL;
    cudaMalloc(&cudaDeviceProbs, sizeof(double) * 4 * MAX_DEPTH);

    unsigned* cudaDeviceOutput = NULL;
    cudaMalloc(&cudaDeviceOutput, sizeof(unsigned) * 2 * nEdges);

    GlobalConstants params;
    // // Initialize parameters in constant memory.  We didn't talk about
    // // constant memory in class, but the use of read-only constant
    // // memory here is an optimization over just sticking these values
    // // in device global memory.  NVIDIA GPUs have a few special tricks
    // // for optimizing access to constant memory.  Using global memory
    // // here would have worked just as well.  See the Programmer's
    // // Guide for more information about constant memory.


    //Generate Probabilities
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    static std::default_random_engine generator;
    double probs[MAX_DEPTH*4];
    for (int i = 0; i < MAX_DEPTH*4; i+=4) {
        double A = RMAT_a * (distribution(generator)+0.5);
        double B = RMAT_b * (distribution(generator)+0.5);
        double C = RMAT_c *(distribution(generator)+0.5);
        double D = (1- (RMAT_a+RMAT_b+RMAT_c)) *(distribution(generator)+0.5);
        double abcd = A+B+C+D;
        probs[i] = A/abcd;
        probs[i+1] = B/abcd;
        probs[i+2] = C/abcd;
        probs[i+3] = D/abcd;
    }
    
    params.cudaDeviceNumEdges = nEdges ;
    params.cudaDeviceNumVertices = nVertices;
    params.cudaDeviceOutput = cudaDeviceOutput;
    params.cudaDeviceProbs = probs;
    cudaMemcpyToSymbol(cuConstGraphParams, &params, sizeof(GlobalConstants));

    return true;
}

bool destroy(){
    cudaFree(cuConstGraphParams.cudaDeviceProbs);
    cudaFree(cuConstGraphParams.cudaDeviceOutput);
    return true;
    // cudaFree(cuConstGraphParams);
}