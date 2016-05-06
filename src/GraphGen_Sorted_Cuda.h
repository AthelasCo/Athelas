#ifndef GRAPH_GEN_SORTED_CUDA_H
#define GRAPH_GEN_SORTED_CUDA_H
#include <fstream>

#define MAX_DEPTH 128
#define NUM_CUDA_THREADS 256
#define NUM_BLOCKS 256
#define THREADS_PER_BLOCK ((NUM_CUDA_THREADS)*(NUM_CUDA_THREADS))
#define SCAN_BLOCK_DIM (THREADS_PER_BLOCK)

struct cudaSquare;
typedef struct curandStateXORWOW curandState_t;

class GraphGen_Sorted_Cuda {
private:
    double* cudaDeviceProbs;
    int* cudaDeviceOutput;
    cudaSquare* cudaDeviceSquares;
    curandState_t* cudaThreadStates;

public:
    GraphGen_Sorted_Cuda();
    int setup(const unsigned long long nEdges,
        const unsigned long long nVertices,
        const double RMAT_a, const double RMAT_b, const double RMAT_c,
        const unsigned long long standardCapacity,
        const bool allowEdgeToSelf,
        const bool allowDuplicateEdges,
        const bool directedGraph,
        const bool sorted
    );
    virtual ~GraphGen_Sorted_Cuda();
    void generate(const bool directedGraph,
        const bool allowEdgeToSelf, const bool sorted, int square_size);
    unsigned long long printGraph(unsigned *Graph, unsigned long long nEdges, std::ofstream& outFile);
    bool destroy();
    void getGraph(unsigned* Graph, unsigned long long nEdges);
};
#endif // GRAPH_GEN_SORTED_CUDA_H

