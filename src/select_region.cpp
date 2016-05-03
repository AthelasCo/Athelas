#include "select_region.hpp"
#include <utility>
#include <boost>

uint32_t SelectRegion::get_codon() {
    uint64_t u = rand_uint64();
    uint32_t x = u&63;
    return (u < h[x]) ? x : a[x];
}

/*
 *  Initialize alias table from discrete distribution, pp
 */

SelectRegion::SelectRegion(const double *pp) {
    // normalize pp and copy into buffer
    double f=0.0, p[64];
    for(int i=0;i<64;++i)
        f += pp[i];
    f = 64.0/f;
    for(int i=0;i<64;++i)
        p[i] = pp[i]*f;
    
    // find starting positions
    std::size_t g,m,mm;
    for(g=0; g<64 && p[g] <  1.0; ++g)
        /*noop*/;
    for(m=0; m<64 && p[m] >= 1.0; ++m)
        /*noop*/;
    mm = m+1;
    
    // build alias table until we run out of large or small bars
    while(g < 64 && m < 64) {
        // convert double to 64-bit integer, control for precision
        h[m] = (static_cast<uint64_t>(
                ceil(p[m]*9007199254740992.0)) << 11);
        a[m] = g;
        p[g] = (p[g]+p[m])-1.0;
        if(p[g] >= 1.0 || mm <= g) {
            for(m=mm;m<64 && p[m] >= 1.0; ++m)
                /*noop*/;
            mm = m+1;
        } else
            m = g;
        for(; g<64 && p[g] <  1.0; ++g)
            /*noop*/;
    }
    
    // any bars that remain have no alias 
    for(; g<64; ++g) {
        if(p[g] < 1.0)
            continue;
        h[g] = std::numeric_limits<boost::uint64_t>::max();
        a[g] = g;
    }
    if(m < 64) {
        q[m] = std::numeric_limits<boost::uint64_t>::max();
        a[m] = m;
        for(m=mm; m<64; ++m) {
            if(p[m] > 1.0)
                continue;
            h[m] = std::numeric_limits<boost::uint64_t>::max();
            a[m] = m;
        }
    }
}


