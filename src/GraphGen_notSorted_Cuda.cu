#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <iostream>
#include <cstring>
#include <fstream>
#include <cstdlib>

#include <curand.h>
#include <curand_kernel.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "GraphGen_notSorted_Cuda.h"
#include "internal_config.hpp"
#include "Square.hpp"
#include "Edge.hpp"
#include "utils.hpp"

struct cudaSquare {
	unsigned long long X_start, X_end, Y_start, Y_end;
	unsigned long long nEdgeToGenerate, level, recIndex_horizontal, recIndex_vertical;
	unsigned long long thisEdgeToGenerate;
};

struct GlobalConstants {

    unsigned long long cudaDeviceNumEdges, cudaDeviceNumVertices;
    double* cudaDeviceProbs;
    int* cudaDeviceOutput;
    cudaSquare* cudaSquares;
    curandState_t* cudaThreadStates;
    int nSquares;
};

__device__ inline int updiv(int n, int d) {
    return (n+d-1)/d;
}

__constant__ GlobalConstants cuConstGraphParams;

/* CUDA's random number library uses curandState_t to keep track of the seed value
   we will store a random state for every thread  */

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed) {

  /* we have to initialize the state */
    // printf("seed %d\n", seed);
  curandState_t* states = cuConstGraphParams.cudaThreadStates;
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x*blockDim.x+threadIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x*blockDim.x+threadIdx.x]);
  // const double RndProb = curand_uniform(states + blockIdx.x);
  // printf("RANDOM RANDOM %lf\n", RndProb);
}

__device__ __inline__ int2
get_Edge_indices(curandState_t* states,  unsigned long long offX, unsigned long long rngX, unsigned long long offY, unsigned long long rngY, double A[],double B[],double C[],double D[]) {
    unsigned long long x_offset = offX, y_offset = offY;
    int depth =0;
    double sumA, sumAB, sumABC, sumAC;
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    curandState_t localState = states[idx];
    while (rngX > 1 || rngY > 1) {
        sumA = A[depth];
        sumAB = sumA + B[depth];
        sumAC = sumA + C[depth];
        sumABC = sumAB + C[depth];
        

        const double RndProb = curand_uniform(&localState);
        // printf("%d %d RANDOM %lf\n", blockIdx.x , threadIdx.x,RndProb );
        if (rngX>1 && rngY>1) {
          if (RndProb < sumA) { rngX/=2; rngY/=2; }
          else if (RndProb < sumAB) { offX+=rngX/2;  rngX-=rngX/2;  rngY/=2; }
          else if (RndProb < sumABC) { offY+=rngY/2;  rngX/=2;  rngY-=rngY/2; }
          else { offX+=rngX/2;  offY+=rngY/2;  rngX-=rngX/2;  rngY-=rngY/2; }
        } else
        if (rngX>1) { // row vector
          if (RndProb < sumAC) { rngX/=2; rngY/=2; }
          else { offX+=rngX/2;  rngX-=rngX/2;  rngY/=2; }
        } else
        if (rngY>1) { // column vector
          if (RndProb < sumAB) { rngX/=2; rngY/=2; }
          else { offY+=rngY/2;  rngX/=2;  rngY-=rngY/2; }
        } else{
            printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
        }
        depth++;
    }
    states[idx] = localState;
    int2 e;
    printf("Edge %d %d\n", (int)offX, (int)offY);

    e.x = offX- x_offset ;
    e.y = offY- y_offset ;
    return e;
}
__global__ void KernelGenerateEdges(const bool directedGraph,
        const bool allowEdgeToSelf, const bool sorted) {
    // std::uniform_int_distribution<>& dis, std::mt19937_64& gen,
    // std::vector<unsigned long long>& duplicate_indices
    curandState_t* states = cuConstGraphParams.cudaThreadStates;
    int blockIndex = blockIdx.x;
    int threadIndex = threadIdx.x;
    printf("BlockIdx %d ThreadIdx %d\n",blockIdx.x, threadIdx.x);
    if (blockIndex < cuConstGraphParams.nSquares) {
        cudaSquare squ = cuConstGraphParams.cudaSquares[blockIndex];
        __shared__ unsigned long long offX;  
        __shared__ unsigned long long offY;  
        __shared__ unsigned long long rngX;  
        __shared__ unsigned long long rngY;  
        
        unsigned long long nEdgesToGen = squ.nEdgeToGenerate;

        __shared__ double A[MAX_DEPTH];
        __shared__ double B[MAX_DEPTH];
        __shared__ double C[MAX_DEPTH];
        __shared__ double D[MAX_DEPTH];

        if (threadIndex==0)
        {
            printf("BlockIdx %d %d\n",blockIdx.x, blockDim.x);
            for (int i = 0; i < MAX_DEPTH; ++i)
            {
                printf("Found Square %d\n", 4*i);
                //double4 prob= *(double4*)(&cuConstGraphParams.cudaDeviceProbs[4 * (i)]);
                //A[i] = prob.x;
                //B[i] = prob.y;
                //C[i] = prob.z;
                //D[i] = prob.w;
                A[i] = (double)(cuConstGraphParams.cudaDeviceProbs[4 * (i)]);
                B[i] = (double)(cuConstGraphParams.cudaDeviceProbs[4 * (i) + 1]);
                C[i] = (double)(cuConstGraphParams.cudaDeviceProbs[4 * (i)+ 2]);
                D[i] = (double)(cuConstGraphParams.cudaDeviceProbs[4 * (i)+ 3]);
                printf("Found Square %d\n", 4*i);
                offX = squ.X_start;
                offY = squ.Y_start;
                rngX = squ.X_start-offX;
                rngY = squ.Y_end-offY;

            }

        }

        auto applyCondition = directedGraph || ( offX < offY); // true: if the graph is directed or in case it is undirected, the square belongs to the lower triangle of adjacency matrix. false: the diagonal passes the rectangle and the graph is undirected.


        unsigned maxIter = updiv(nEdgesToGen, blockDim.x);

        for (unsigned i = 0; i < maxIter; ++i)
        {
            int edgeIdx = i * blockDim.x + threadIndex;
            int2 e;
            if (edgeIdx < nEdgesToGen )
            {

                while(true) {
                    printf("Starting Edge\n");
                    e = get_Edge_indices(states, offX, rngX, offY, rngY, A, B, C, D );
                    unsigned long long h_idx = e.x+offX;
                    unsigned long long v_idx = e.y+offY;
                    if( (!applyCondition && h_idx > v_idx) || (!allowEdgeToSelf && h_idx == v_idx ) ) // Short-circuit if it doesn't pass the test.
                        continue;
                    if (h_idx< offX || h_idx>= offX+rngX || v_idx < offY || v_idx >= offY+rngY ){
                        printf(" recompute\n" );
                        continue;
                    }
                    break;
                }
                *(int2*)cuConstGraphParams.cudaDeviceOutput[2*( squ.thisEdgeToGenerate + i*blockDim.x + threadIndex )] = e;

            }
            __syncthreads();
        }
        __syncthreads();
    }

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

GraphGen_notSorted_Cuda::GraphGen_notSorted_Cuda() {
    cudaDeviceProbs = NULL;
    cudaDeviceOutput = NULL;
    cudaDeviceSquares = NULL;
}


GraphGen_notSorted_Cuda::~GraphGen_notSorted_Cuda() {
    if (cudaDeviceProbs) {
        cudaFree(cudaDeviceProbs);
        cudaFree(cudaDeviceOutput);
        cudaFree(cudaDeviceSquares);
        cudaFree(cudaThreadStates);
   }
}

int GraphGen_notSorted_Cuda::setup(
        const unsigned long long nEdges,
        const unsigned long long nVertices,
        const double RMAT_a, const double RMAT_b, const double RMAT_c,
        const unsigned long long standardCapacity,
        const bool allowEdgeToSelf,
        const bool allowDuplicateEdges,
        const bool directedGraph,
        const bool sorted
    ){
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
    cudaMalloc(&cudaDeviceProbs, sizeof(double) * 4 * MAX_DEPTH);




    cudaMalloc(&cudaDeviceOutput, sizeof(int) * 2 * nEdges);

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
    cudaMemcpy(cudaDeviceProbs, probs, sizeof(double) * 4 * MAX_DEPTH, cudaMemcpyHostToDevice);
    params.cudaDeviceProbs = cudaDeviceProbs;

    //Generate Squares
    std::vector<Square> squares ( 1, Square( 0, nVertices, 0, nVertices, nEdges, 0, 0, 0 ) );
	bool allRecsAreInRange;
	do {
		allRecsAreInRange = true;

		unsigned int recIdx = 0;
		for( auto& rec: squares ) {

			if( Eligible_RNG_Rec(rec, standardCapacity) ) {
				// continue;
			} else {
				ShatterSquare(squares, RMAT_a, RMAT_b, RMAT_c, recIdx, directedGraph);
				allRecsAreInRange = false;
				
				break;
			}
			++recIdx;
		}
	} while( !allRecsAreInRange );

	// Making sure there are enough squares to utilize all blocks and not more
	while( squares.size() < NUM_BLOCKS && !edgeOverflow(squares) ) {
		// Shattering the biggest rectangle.
		unsigned long long biggest_size = 0;
		unsigned int biggest_index = 0;
		for( unsigned int x = 0; x < squares.size(); ++x )
			if( squares.at(x).getnEdges() > biggest_size ) {
				biggest_size = squares.at(x).getnEdges();
				biggest_index = x;
			}
		ShatterSquare(squares, RMAT_a, RMAT_b, RMAT_c, biggest_index, directedGraph);
	}

	if (allowDuplicateEdges)
	{
		int originalSize = squares.size();
		for (int index = 0; index < originalSize; ++index)
		{
			//memory leak?
			Square srcRect(squares.at(index));
			// squares.erase(squares.begin()+index);
		
			int numEdgesAssigned = 0;
			int edgesPerSquare = srcRect.getnEdges()/NUM_BLOCKS;
			if (edgesPerSquare<20000)
			{
				continue;
			}
			for( unsigned int i = 0; i < NUM_BLOCKS-1; ++i ){
				Square destRect(srcRect);
				destRect.setnEdges(edgesPerSquare);
				numEdgesAssigned+=edgesPerSquare;
				squares.push_back(destRect);

			}
			srcRect.setnEdges( srcRect.getnEdges()-numEdgesAssigned);
			squares.at(index) = srcRect;
		}

	
	}
	std::sort(squares.begin(), squares.end(),std::greater<Square>());

    //unsigned long long* allSquares = (unsigned long long*) malloc(sizeof(unsigned long long)* 9 * squares.size());
    cudaSquare* allSquares = (cudaSquare*) malloc(sizeof(cudaSquare) * squares.size());

    unsigned long long tEdges = 0;

    for( unsigned int x = 0; x < squares.size(); ++x ) {
		Square& rec = squares.at( x );
        cudaSquare newSquare;
        newSquare.X_start = rec.get_X_start();
        newSquare.X_end = rec.get_X_end();
        newSquare.Y_start = rec.get_Y_start();
        newSquare.Y_end = rec.get_Y_end();
        newSquare.nEdgeToGenerate = rec.getnEdges();
        newSquare.level = 0;//TODO
        newSquare.recIndex_horizontal = rec.get_H_idx();
        newSquare.recIndex_vertical = rec.get_V_idx();
        newSquare.thisEdgeToGenerate = tEdges;
        memcpy(allSquares+x, &newSquare, sizeof(cudaSquare));
        tEdges += rec.getnEdges();
    }
    cudaMalloc(&cudaDeviceSquares, sizeof(cudaSquare) * squares.size());
    cudaMemcpy(cudaDeviceSquares, allSquares, sizeof(cudaSquare) * squares.size(), cudaMemcpyHostToDevice);
    params.cudaSquares = cudaDeviceSquares;
    params.nSquares = squares.size();

    /* allocate space on the GPU for the random states */
    cudaMalloc((void**) &cudaThreadStates, squares.size()*NUM_CUDA_THREADS * sizeof(curandState_t));
    params.cudaThreadStates = cudaThreadStates;

    cudaMemcpyToSymbol(cuConstGraphParams, &params, sizeof(GlobalConstants));
    /* invoke the GPU to initialize all of the random states */
    init<<<squares.size(), NUM_CUDA_THREADS>>>(time(0));
    cudaDeviceSynchronize();

    for( unsigned int x = 0; x < squares.size(); ++x )
        std::cout << squares.at(x);
    std::cout << "CUDA Error " << cudaGetErrorString(cudaGetLastError());
    return squares.size();
}

void GraphGen_notSorted_Cuda::generate(const bool directedGraph,
        const bool allowEdgeToSelf, const bool sorted, int squares_size) {
    dim3 blockDim(NUM_CUDA_THREADS);
    // dim3 gridDim(updivHost(squares_size, blockDim.x));
    dim3 gridDim(squares_size);
    printf("Hello \n");
    KernelGenerateEdges<<<gridDim, blockDim>>>(directedGraph,
        allowEdgeToSelf, sorted);
    cudaDeviceSynchronize();
    std::cout << "CUDA Error " << cudaGetErrorString(cudaGetLastError());
    
    printf("Bye \n");

}

void GraphGen_notSorted_Cuda::printGraph(unsigned *Graph, unsigned long long nEdges, std::ofstream& outFile) {
    // for (unsigned long long x = 0; x < nEdges; x++) {
    //     outFile << Graph[2*x] << "\t" << Graph[2*x+1] << "\n";
    // }
}

bool GraphGen_notSorted_Cuda::destroy(){
    //cudaFree(states);
    cudaFree(cudaDeviceProbs);
    cudaFree(cudaDeviceOutput);
    return true;
}

void GraphGen_notSorted_Cuda::getGraph(unsigned* Graph, unsigned long long nEdges) {
     // cudaMemcpy(Graph, cudaDeviceOutput, sizeof(int)*2*nEdges, cudaMemcpyDeviceToHost);
}

