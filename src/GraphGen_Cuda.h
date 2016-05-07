#ifndef GRAPH_GEN_SORTED_CUDA_H
#define GRAPH_GEN_SORTED_CUDA_H
#include <fstream>

#define MAX_DEPTH 128
#define NUM_CUDA_THREADS 256
#define NUM_BLOCKS 1024
#define THREADS_PER_BLOCK ((NUM_CUDA_THREADS)*(NUM_CUDA_THREADS))
#define SCAN_BLOCK_DIM (THREADS_PER_BLOCK)
#define MAX_NUM_EDGES_PER_BLOCK ((NUM_CUDA_THREADS)*20)

struct cudaSquare;
typedef struct curandStateXORWOW curandState_t;

class GraphGen_Cuda {
private:
    double* cudaDeviceProbs;
    int* cudaDeviceOutput;
    uint* cudaDeviceCompressedOutput;
    cudaSquare* cudaDeviceSquares;
    curandState_t* cudaThreadStates;
    bool compressed;
    cudaSquare* allSquares;
    uint nSquares;

public:
    GraphGen_Cuda();
    int setup(const uint nEdges,
        const uint nVertices,
        const double RMAT_a, const double RMAT_b, const double RMAT_c,
        const uint standardCapacity,
        const bool allowEdgeToSelf,
        const bool allowDuplicateEdges,
        const bool directedGraph,
        const bool sorted,
        const bool compressed
    );
    virtual ~GraphGen_Cuda();
    void generate(const bool directedGraph,
        const bool allowEdgeToSelf, const bool sorted, int square_size);
    uint printGraph(unsigned *Graph, uint nEdges, std::ofstream& outFile);
    bool destroy();
    void getGraph(unsigned* Graph, uint nEdges);
};
#endif // GRAPH_GEN_SORTED_CUDA_H


