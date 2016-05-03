#ifndef GRAPH_GEN_NOT_SORTED_CUDA_H
#define GRAPH_GEN_NOT_SORTED_CUDA_H
#include <fstream>

#define MAX_DEPTH 128
#define NUM_CUDA_THREADS 32
#define THREADS_PER_BLOCK ((NUM_CUDA_THREADS)*(NUM_CUDA_THREADS))
#define SCAN_BLOCK_DIM (THREADS_PER_BLOCK)

bool setup(const unsigned long long nEdges,
        const unsigned long long nVertices,
        const double RMAT_a, const double RMAT_b, const double RMAT_c,
        const unsigned int nCPUWorkerThreads,
        std::ofstream& outFile,
        const unsigned long long standardCapacity,
        const bool allowEdgeToSelf,
        const bool allowDuplicateEdges,
        const bool directedGraph,
        const bool sorted);

#endif // GRAPH_GEN_NOT_SORTED_CUDA_H


