#ifndef SELECT_REGION_HPP
#define SELECT_REGION_HPP


#include "internal_config.hpp"
#include <utility>
#include <stdint.h>




class SelectRegion{

protected:
    // Two members standing for source vertex index and destination vertex index.
    // The type of these two member variables can be found at internal_config.hpp.
    uint32_t a[64]; //aliases
    uint64_t h[64]; //bar heights

public:
    ~SelectRegion(){}
    SelectRegion(const double *pp);

    uint32_t get_codon(uint64_t u);
};

#endif  // SELECT_REGION_HPP
