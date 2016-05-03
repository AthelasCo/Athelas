#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

__device__ inline int updiv(int n, int d) {
    return (n+d-1)/d;
}


static inline int updivHost(int n, int d) {
    return (n+d-1)/d;
}

struct GlobalConstants {

    double* cudaConstantProbTable;
    unsigned long long cudaDeviceNumEdges, cudaDeviceNumVertices;
};

__constant__ GlobalConstants cuConstGraphParams;


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
    double* cudaConstantProbTable;
    static std::default_random_engine generator;

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy
    cudaMalloc(&cudaConstantProbTable , sizeof(double) * 128 * 4 );
    // cudaMemcpy(&cudaDeviceNumEdges, sizeof(unsigned long long));
    // cudaMemcpy(&cudaDeviceNumVertices , sizeof(unsigned long long));


    GlobalConstants params;
    params.cudaConstantProbTable = cudaConstantProbTable;
    params.cudaDeviceNumEdges = nEdges ;
    params.cudaDeviceNumVertices = nVertices;
    cudaMemcpyToSymbol(cuConstGraphParams, &params, sizeof(GlobalConstants));
    // cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    // cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    // cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    // cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    // cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);
    // cudaMalloc(&cudaDeviceCircleList, sizeof(float) * (numCircles+THREADS_PER_BLOCK) * updivHost(image->width,NUM_THREADS) * updivHost(image->height,NUM_THREADS));

    // cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    // cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    // cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    // cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // cudaMemset(cudaDeviceCircleList,0.f, sizeof(float) * numCircles * updivHost(image->width,NUM_THREADS) * updivHost(image->height,NUM_THREADS));

    // // Initialize parameters in constant memory.  We didn't talk about
    // // constant memory in class, but the use of read-only constant
    // // memory here is an optimization over just sticking these values
    // // in device global memory.  NVIDIA GPUs have a few special tricks
    // // for optimizing access to constant memory.  Using global memory
    // // here would have worked just as well.  See the Programmer's
    // // Guide for more information about constant memory.

    // GlobalConstants params;
    // params.sceneName = sceneName;
    // params.numCircles = numCircles;
    // params.imageWidth = image->width;
    // params.imageHeight = image->height;
    // params.position = cudaDevicePosition;
    // params.velocity = cudaDeviceVelocity;
    // params.color = cudaDeviceColor;
    // params.radius = cudaDeviceRadius;
    // params.imageData = cudaDeviceImageData;
    // params.circleList = cudaDeviceCircleList;

    // cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // // also need to copy over the noise lookup tables, so we can
    // // implement noise on the GPU
    // int* permX;
    // int* permY;
    // float* value1D;
    // getNoiseTables(&permX, &permY, &value1D);
    // cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    // cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    // cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // // last, copy over the color table that's used by the shading
    // // function for circles in the snowflake demo

    // float lookupTable[COLOR_MAP_SIZE][3] = {
    //     {1.f, 1.f, 1.f},
    //     {1.f, 1.f, 1.f},
    //     {.8f, .9f, 1.f},
    //     {.8f, .9f, 1.f},
    //     {.8f, 0.8f, 1.f},
    // };

    // cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);
    return true;
}
