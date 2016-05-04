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
    bool directedGraph, allowEdgeToSelf, sorted;
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
    //printf("Edge %d %d\n", (int)offX, (int)offY);

    e.x = offX;
    e.y = offY;
    return e;
}
__global__ void KernelGenerateEdges() {
    // std::uniform_int_distribution<>& dis, std::mt19937_64& gen,
    // std::vector<unsigned long long>& duplicate_indices
    //printf("BlockIdx %d ThreadIdx %d\n",blockIdx.x, threadIdx.x);
    curandState_t* states = cuConstGraphParams.cudaThreadStates;
    bool directedGraph = cuConstGraphParams.directedGraph;
    bool allowEdgeToSelf = cuConstGraphParams.allowEdgeToSelf;
    bool sorted = cuConstGraphParams.sorted;
    int blockIndex = blockIdx.x+blockIdx.y;
    int threadIndex = threadIdx.x;
    if (blockIndex < cuConstGraphParams.nSquares) {
        cudaSquare squ = cuConstGraphParams.cudaSquares[blockIndex];
        __shared__ unsigned long long offX;  
        __shared__ unsigned long long offY;  
        __shared__ unsigned long long rngX;  
        __shared__ unsigned long long rngY;  
        
        __shared__ unsigned long long nEdgesToGen;
        if (threadIndex==0)
        {
            offX = squ.X_start;
            offY = squ.Y_start;
            rngX = squ.X_end-offX;
            rngY = squ.Y_end-offY;
            nEdgesToGen = squ.nEdgeToGenerate;
            printf("Found Square %d with tl %d tr %d bl %d br %d and edges %d for hidx %d vid %d tE %d\n", blockIndex, offX, offY, offX+rngX, offY+rngY, nEdgesToGen,
                                        squ.recIndex_horizontal, squ.recIndex_vertical, squ.thisEdgeToGenerate);
        }   
        __shared__ double A[MAX_DEPTH];
        __shared__ double B[MAX_DEPTH];
        __shared__ double C[MAX_DEPTH];
        __shared__ double D[MAX_DEPTH];

        if (threadIndex==0)
        {
            for (int i = 0; i < MAX_DEPTH; ++i)
            {
                A[i] = (double)(cuConstGraphParams.cudaDeviceProbs[4 * (i)]);
                B[i] = (double)(cuConstGraphParams.cudaDeviceProbs[4 * (i) + 1]);
                C[i] = (double)(cuConstGraphParams.cudaDeviceProbs[4 * (i)+ 2]);
                D[i] = (double)(cuConstGraphParams.cudaDeviceProbs[4 * (i)+ 3]);
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
                    e = get_Edge_indices(states, offX, rngX, offY, rngY, A, B, C, D );
                    unsigned long long h_idx = e.x;
                    unsigned long long v_idx = e.y;
                    if( (!applyCondition && h_idx > v_idx) || (!allowEdgeToSelf && h_idx == v_idx ) ) {// Short-circuit if it doesn't pass the test.
                        printf("EdgeID %d fail1\n", edgeIdx );
                        continue;
                    } else if (h_idx< offX || h_idx>= offX+rngX || v_idx < offY || v_idx >= offY+rngY ){
                        printf("EdgeID %d recompute src %d dst %d tl %d tr %d bl %d br %d \n", edgeIdx, h_idx, v_idx, offX, offY, offX+rngX, offY+rngY);
                        continue;
                    } else {
                        break;
                    }
                }
                printf("Edges Calculated %d \t %d\n", e.x,e.y);
                cuConstGraphParams.cudaDeviceOutput[2*( squ.thisEdgeToGenerate + edgeIdx)] = e.x;
                cuConstGraphParams.cudaDeviceOutput[2*( squ.thisEdgeToGenerate + edgeIdx)+1] = e.y;

            }
            __syncthreads();
        }
        __syncthreads();
    }

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
    params.allowEdgeToSelf = allowEdgeToSelf;
    params.directedGraph = directedGraph;
    params.sorted = sorted;
    cudaMemcpyToSymbol(cuConstGraphParams, &params, sizeof(GlobalConstants));
    /* invoke the GPU to initialize all of the random states */
    init<<<squares.size(), NUM_CUDA_THREADS>>>(time(0));
    cudaDeviceSynchronize();

    for( unsigned int x = 0; x < squares.size(); ++x ){
        std::cout << squares.at(x);
        std::cout << "Edges to gen : " << allSquares[x].thisEdgeToGenerate << "\n";
    }
    std::cout << "CUDA Error " << cudaGetErrorString(cudaGetLastError()) << "\n";
    return squares.size();
}

void GraphGen_notSorted_Cuda::generate(const bool directedGraph,
        const bool allowEdgeToSelf, const bool sorted, int squares_size) {
    dim3 nThreads(NUM_CUDA_THREADS,1,1);
    // dim3 gridDim(updivHost(squares_size, blockDim.x));
    dim3 nBlocks(squares_size,1,1);
    printf("Hello launching kernel of blocks %d %d %d and tpb %d %d %d\n", nBlocks.x, nBlocks.y, nBlocks.z, nThreads.x, nThreads.y, nThreads.z);
    KernelGenerateEdges<<<nBlocks, nThreads>>>();
    cudaDeviceSynchronize();
    std::cout << "CUDA Error " << cudaGetErrorString(cudaGetLastError());
    
    printf("Bye \n");

}

void GraphGen_notSorted_Cuda::printGraph(unsigned *Graph, unsigned long long nEdges, std::ofstream& outFile) {
    for (unsigned long long x = 0; x < nEdges; x++) {
         outFile << Graph[2*x] << "\t" << Graph[2*x+1] << "\n";
    }
}

bool GraphGen_notSorted_Cuda::destroy(){
    //cudaFree(states);
    cudaFree(cudaDeviceProbs);
    cudaFree(cudaDeviceOutput);
    return true;
}

void GraphGen_notSorted_Cuda::getGraph(unsigned* Graph, unsigned long long nEdges) {
     cudaMemcpy(Graph, cudaDeviceOutput, sizeof(int)*2*nEdges, cudaMemcpyDeviceToHost);
}

