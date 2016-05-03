#include <cmath> 		// For std::floor.
#include <ctime>		// For std::time.
#include <cstdlib>		// For std::rand.
#include <functional>	// For std::ref.
#include <iostream>		// For std::cout.

#if defined(_WIN32)
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "utils.hpp"
#include "internal_config.hpp"
#include <random>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <ctime>

template <class T>
double gen_uniform_normalized_double(T& random)
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(random);
}

template <class T>
class alias_method
{
public:
    /**
     * Constructs a new alias_method to sample from a discrete distribution and
     * hand back outcomes based on the probability distribution.
     * <p>
     * Given as input a list of probabilities corresponding to outcomes 0, 1,
     * ..., n - 1, along with the random number generator that should be used
     * as the underlying generator, this constructor creates the probability 
     * and alias tables needed to efficiently sample from this distribution.
     *
     * @param probabilities The list of probabilities.
     * @param random The random number generator
     */
    alias_method(const std::vector<double>& probability, T& random)
        : probability_(probability), random_(random)
    {
    // Allocate space for the alias table.
        alias_.resize(probability_.size());

        // Compute the average probability and cache it for later use.
        const double average = 1.0 / probability_.size();

        // two stacks to act as worklists as we populate the tables
        std::vector<int> small, large;

        // Populate the stacks with the input probabilities.
        for(size_t i=0; i<probability_.size(); ++i)
        {
            // If the probability is below the average probability, then we add
            // it to the small list; otherwise we add it to the large list.
            if (probability_[i] >= average)
                large.push_back(i);
            else
                small.push_back(i);
        }

        // As a note: in the mathematical specification of the algorithm, we
        // will always exhaust the small list before the big list.  However,
        // due to floating point inaccuracies, this is not necessarily true.
        // Consequently, this inner loop (which tries to pair small and large
        // elements) will have to check that both lists aren't empty.
        while(!small.empty() && !large.empty()) 
        {
            // Get the index of the small and the large probabilities.
            int less = small.back(); small.pop_back();
            int more = large.back(); large.pop_back();

            alias_[less] = more;

            // Decrease the probability of the larger one by the appropriate
            // amount.
            probability_[more] = (probability_[more] + probability_[less]) - average;

            // If the new probability is less than the average, add it into the
            // small list; otherwise add it to the large list.
            if (probability_[more] >= average)
                large.push_back(more);
            else
                small.push_back(more);
        }

        // At this point, everything is in one list, which means that the
        // remaining probabilities should all be 1/n.  Based on this, set them
        // appropriately.  Due to numerical issues, we can't be sure which
        // stack will hold the entries, so we empty both.
        while(!small.empty())
        {
            probability_[small.back()] = average;
            small.pop_back();
        }
        while(!large.empty())
        {
            probability_[large.back()] = average;
            large.pop_back();
        }
        
        // These probabilities have not yet been scaled up to be such that
        // 1/n is given weight 1.0.  We do this here instead.
        int n = static_cast<int>(probability_.size());
        std::transform(probability_.cbegin(), probability_.cend(), probability_.begin(), 
            [n](double p){ return p * n; });
    }

    /**
     * Samples a value from the underlying distribution.
     *
     * @return A random value sampled from the underlying distribution.
     */
    int next()
    {
        // Generate a fair die roll to determine which column to inspect.
        int column = random_() % probability_.size();
        // Generate a biased coin toss to determine which option to pick.
        bool coinToss = gen_uniform_normalized_double(random_) < probability_[column];

        // Based on the outcome, return either the column or its alias.
        return coinToss? column : alias_[column];
    }

private:
  // The probability and alias tables.
    std::vector<int> alias_;
    std::vector<double> probability_;
  // The random number generator used to sample from the distribution.
    T& random_;
};

template <class T>
alias_method<T> make_alias_method(const std::vector<double>& probabilities, T& random)
{
    return alias_method<T>(probabilities, random);
}

unsigned long long calculateAvailableRAM( const unsigned long long totalRAM, const double memUse ) {
	return static_cast<unsigned long long>( static_cast<double>(totalRAM)*memUse );
}

size_t getTotalSystemMemory() {
#if defined(_WIN32)
	MEMORYSTATUSEX status;
	status.dwLength = sizeof(status);
	GlobalMemoryStatusEx(&status);
	return status.ullTotalPhys;
#else
	long pages = sysconf(_SC_PHYS_PAGES);
	long page_size = sysconf(_SC_PAGE_SIZE);
	return pages * page_size;
#endif
}

bool Eligible_RNG_Squares( std::vector<Square>& squares, const unsigned long long nEdgesThreshold ) {
	unsigned long long nEdgesInEachColumn = 0;	// Accumulated number of edges from squares having the same horizontal coordination.
	unsigned long long beingExamined_X_end = 0;
	for( auto& rec: squares ) {
		if( beingExamined_X_end == rec.get_X_end() ) {
			nEdgesInEachColumn += rec.getnEdges();
		}
		else {
			nEdgesInEachColumn = rec.getnEdges();
			beingExamined_X_end = rec.get_X_end();
		}
		if( nEdgesInEachColumn > nEdgesThreshold )
			return false;
	}

	return true;
}

unsigned long long Get_N_Columns( std::vector<Square>& squares ) {
	unsigned long long nColumns = 0;
	unsigned long long beingExamined_X_end = 0;
	for( auto& rec: squares )
		if( beingExamined_X_end != rec.get_X_end() ) {
			beingExamined_X_end = rec.get_X_end();
			++nColumns;
		}
	return nColumns;
}

void ShatterSquare( std::vector<Square>& square, const double a, const double b, const double c, const unsigned int index, const bool directedGraph ) {

	std::srand(std::time(0));

	// Noise for selecting a, b, c to cut square.
	auto noise_a = static_cast<double>(std::rand()-(RAND_MAX/2))/(RAND_MAX*100.0);	// Very small random noises. Maybe can be implemented in a better way. Up to one percent for each parameter.
	auto noise_b = static_cast<double>(std::rand()-(RAND_MAX/2))/(RAND_MAX*100.0);
	auto noise_c = static_cast<double>(std::rand()-(RAND_MAX/2))/(RAND_MAX*100.0);

	// Noise for number of edges to be created by each sub-square.
	auto noise_a_edge_share = static_cast<double>(std::rand()-(RAND_MAX/2))/(RAND_MAX*100.0);	// Very small random noises. Maybe can be implemented in a better way. Up to one percent for each parameter.
	auto noise_b_edge_share = static_cast<double>(std::rand()-(RAND_MAX/2))/(RAND_MAX*100.0);
	auto noise_c_edge_share = static_cast<double>(std::rand()-(RAND_MAX/2))/(RAND_MAX*100.0);

	Square srcRect(square.at(index));
	square.erase(square.begin()+index);

	// Create and push 4 resulted square from the shattered one.
	for( unsigned int i = 0; i < 4; ++i )
		square.push_back(srcRect.Get_part( i, noise_a+a, noise_b+b, noise_c+c, noise_a_edge_share+a, noise_b_edge_share+b, noise_c_edge_share+c ));

	// Renormalizing: making sure the number of edges still matches, otherwise multiplications with double may impact final total number of edges.
	square.at(square.size()-1).setnEdges( 	srcRect.getnEdges() -
													square.at(square.size()-2).getnEdges() -
													square.at(square.size()-3).getnEdges() -
													square.at(square.size()-4).getnEdges() );

	if( !directedGraph ) {
		if( srcRect.get_X_end() == srcRect.get_Y_end() ) {	// If the source rectangle is on the main diagonal, we should throw away the 2nd part (b).
			square.at(square.size()-2).setnEdges( 	square.at(square.size()-2).getnEdges() +
															square.at(square.size()-3).getnEdges() );	// We add the edges from b part to c part.
			square.erase(square.end()-3);	// erase the b part from the vector.
		}
	}

}
std::pair<unsigned long long, unsigned long long> get_Edge_indices_PKSG( unsigned long long offX, unsigned long long rngX,unsigned long long offY, unsigned long long rngY, std::uniform_real_distribution<>& distribution, std::default_random_engine& generator , double A[], double B[], double C[], double D[], unsigned long long u, int k) {
  unsigned long long z=u, v=0, s=0;
    // int depth =0;
    for (int depth = 0; depth < k; ++depth)
    {
      double sumAB = A[depth] +B[depth];
      double a = A[depth]/sumAB;
      double b = B[depth]/sumAB;
      double c = C[depth]/(1-sumAB);
      double d = D[depth]/(1-sumAB);
      unsigned long long l = z%2;
      const double RndProb = distribution(generator);
      if (l==0)
      {
        s=1;
        if (RndProb<a)
        {
          s=0;
        }
      }else {
        s=1;
        if (RndProb<c)
        {
          s=0;   
        }
      }
      v= 2*v+s;
      z= z/2;

    }

      return  std::make_pair (u, v);
    
}
std::pair<unsigned long long, unsigned long long> get_Edge_indices( unsigned long long offX, unsigned long long rngX,unsigned long long offY, unsigned long long rngY, const double a, const double b,const double c,std::uniform_real_distribution<>& distribution, std::default_random_engine& generator , double sumA[], double sumAB[], double sumAC[], double sumABC[]) {
	// std::vector<double> probabilities;
 //    probabilities.push_back(a);
 //    probabilities.push_back(b);
 //    probabilities.push_back(c);
 //    probabilities.push_back(1-(a+b+c));
 //    std::mt19937_64 rnd;
 //    rnd.seed(clock());
 //    auto am = make_alias_method(probabilities, rnd);
	// double abcd = a+b+c+d;
 //    double sumA = a/(abcd);
 //    double sumAB = (a+b)/abcd;
 //    double sumAC = (a+c)/abcd;
 //    double sumABC = (a+b+c)/abcd;
    // static std::default_random_engine generator;

    // std::uniform_real_distribution<double> distribution(0.0,1.0);
		// double d = 1-(a+b+c);
  //   for (int i = 0; i < 128; i++) {
  //   const double a = A * (Rnd.GetUniDev() + 0.5);
  //   const double b = B * (Rnd.GetUniDev() + 0.5);
  //   const double c = C * (Rnd.GetUniDev() + 0.5);
  //   const double d = (1.0 - (A+B+C)) * (Rnd.GetUniDev() + 0.5);
  //   const double abcd = a+b+c+d;
  //   sumA.Add(a / abcd);
  //   sumAB.Add((a+b) / abcd);
  //   sumAC.Add((a+c) / abcd);
  //   sumABC.Add((a+b+c) / abcd);
  // }
      // rngX = Nodes;  rngY = Nodes;  offX = startIdx_ull;  offY = 0;
      // Depth = 0;
      // recurse the matrix
    // std::cout <<"FUCK OFF";
    // throw std::invalid_argument( "received negative value" );
    // std::cout<<"here\n";
    int depth =0;
      while (rngX > 1 || rngY > 1) {
    //     double A = a * (distribution(generator)+0.5);
    //     double B = b * (distribution(generator)+0.5);
    //     double C = c *(distribution(generator)+0.5);
    //     double D = d *(distribution(generator)+0.5);
  		// double abcd = A+B+C+D;
  	 //    double sumA[depth] = A/(abcd);
  	 //    double sumAB[] = (A+B)/abcd;
  	 //    double sumAC = (A+C)/abcd;
  	 //    double sumABC = (A+B+C)/abcd;
        const double RndProb = distribution(generator);

        if (rngX>1 && rngY>1) {
          if (RndProb < sumA[depth]) { rngX/=2; rngY/=2; }
          else if (RndProb < sumAB[depth]) { offX+=rngX/2;  rngX-=rngX/2;  rngY/=2; }
          else if (RndProb < sumABC[depth]) { offY+=rngY/2;  rngX/=2;  rngY-=rngY/2; }
          else { offX+=rngX/2;  offY+=rngY/2;  rngX-=rngX/2;  rngY-=rngY/2; }
        } else
        if (rngX>1) { // row vector
          if (RndProb < sumAC[depth]) { rngX/=2; rngY/=2; }
          else { offX+=rngX/2;  rngX-=rngX/2;  rngY/=2; }
        } else
        if (rngY>1) { // column vector
          if (RndProb < sumAB[depth]) { rngX/=2; rngY/=2; }
          else { offY+=rngY/2;  rngX/=2;  rngY-=rngY/2; }
        } else{
    	        throw std::invalid_argument( "received negative value" );

        }
        depth++;
      }

     	return 	std::make_pair (offX, offY);
    
}


unsigned long long genEdgeIndex_FP( unsigned long long startIdx_ull, unsigned long long endIdx_ull, const double a, const double b_or_c, std::uniform_int_distribution<>& dis, std::mt19937_64& gen ) {

	double noise_a = 0, noise_b_or_c = 0, cutLine;
	double cutIndex;
	auto startIdx = static_cast<double>(startIdx_ull);
	auto endIdx = static_cast<double>(endIdx_ull);


	while( (endIdx - startIdx) >= 1.0 ) {
	#ifdef ADD_NOISE_TO_RMAT_PARAMETERS_AT_EACH_LEVEL
			noise_a = static_cast<double>(dis(gen)-(dis.max()/2.0))/(dis.max()*500.0);	// Much smaller noise. Maybe can be improved.
			noise_b_or_c = static_cast<double>(dis(gen)-(dis.max()/2.0))/(dis.max()*500.0);	// Much smaller noise. Maybe can be improved.
	#endif
		cutLine = a + noise_a + b_or_c + noise_b_or_c ;
		cutIndex = (endIdx+startIdx)/2.0;
		if( (static_cast<double>(dis(gen))/dis.max()) < cutLine )
			endIdx = cutIndex;
		else
			startIdx = cutIndex;
	}
	// std::cout <<"WOOOHOOO";
	return static_cast<long long>( std::floor((startIdx+endIdx)/2.0 + 0.5) );

}

void printEdgeGroupNoFlush( std::vector<Edge>& edges, std::ofstream& outFile ) {
	for( auto edge: edges )
		outFile << edge;
}

void printEdgeGroup( std::vector<Edge>& edges, std::ofstream& outFile ) {
	printEdgeGroupNoFlush( edges, outFile );
	outFile.flush();
}

// Checks if there are squares that have to create lots of edges compared to their size and may never finish.
bool edgeOverflow( std::vector<Square>& sqaures ) {
	for( auto& rec : sqaures)
		if( 3*rec.getnEdges() >= (rec.get_X_end()-rec.get_X_start())*(rec.get_Y_end()-rec.get_Y_start()) )	// 3 is experimental. It tells if the size of square is less than 3 times the number of edges, square shouldn't be shattered.
			return true;

	return false;
}
void generate_edges_PSKG( Square& squ,
    std::vector<Edge>& edgesVec,
    const double RMAT_a, const double RMAT_b, const double RMAT_c,
    const bool directedGraph,
    const bool allowEdgeToSelf,
    std::uniform_int_distribution<>& dis, std::mt19937_64& gen,
    std::vector<unsigned long long>& duplicate_indices ) {
  static std::default_random_engine generator;

  std::uniform_real_distribution<double> distribution(0.0,1.0);
  int k=  ceil(log2(squ.get_X_end()-squ.get_X_start()));
  auto applyCondition = directedGraph || ( squ.get_H_idx() < squ.get_V_idx() ); // true: if the graph is directed or in case it is undirected, the square belongs to the lower triangle of adjacency matrix. false: the diagonal passes the rectangle and the graph is undirected.
  auto createNewEdges = duplicate_indices.empty();
  unsigned long long nEdgesToGen = createNewEdges ? squ.getnEdges() : duplicate_indices.size();
  // std::cout<<squ<<createNewEdges<<std::endl;
  double a[k], b[k], c[k], d[k], sumAB[k];
  for (int i = 0; i < k; ++i)
  {

    double A = RMAT_a * (distribution(generator)+0.5);
    double B = RMAT_b * (distribution(generator)+0.5);
    double C = RMAT_c *(distribution(generator)+0.5);
    double D = (1- (RMAT_a+RMAT_b+RMAT_c)) *(distribution(generator)+0.5);
    double abcd = A+B+C+D;
    a[i] = A/abcd;
    b[i] = B/abcd;
    c[i] = C/abcd;
    d[i] = D/abcd;
    sumAB[i] = (a[i]+b[i]);


  }
  unsigned long long numEdges = 0;
  // for Each vertex u do
  // //Determine out-degree of u
  // p = 1; z = u
  // for j = 1, · · · k do
  // l = mod(z, n); p = pUl
  // z = z/n (integer division)
  // end for
  int N=2;
  for (unsigned long long  u = squ.get_X_start(); u < squ.get_X_end(); ++u)
  {
    double p=nEdgesToGen;
    unsigned long long z = u;
    unsigned long long  rngX = squ.get_X_end()-squ.get_X_start();
    int j=0;
    while(rngX>0) {

        unsigned long long l = z%N;
        double Ul = sumAB[j];
        if (l==1)
        {
          Ul = 1-sumAB[j];
        }
        p= p * Ul;
        z = z/N;
        rngX/=2;
        // std::cout<<p<<"p  rngX"<<rngX <<"\n";
        j++;
    }

    double ep =p;
    std::poisson_distribution<unsigned long long > distribution2(ep );
    unsigned long long X = distribution2(gen);
    // std::cout<< "\n"<<X<<" edgesGenerated "<<u<<" "<< ep<<" "<< sumAB[0]<<" " <<numEdges <<"\n";

          
    for( unsigned long long edgeIdx = 0; edgeIdx < X && numEdges < nEdgesToGen; ) {
      unsigned long long h_idx, v_idx;
      std::pair <long long int,long long int> e;
      unsigned long long offX, offY, rngX, rngY;
      offX = squ.get_X_start();
      rngX = squ.get_X_end()-offX;
      offY = squ.get_Y_start();
      rngY = squ.get_Y_end()-offY;
      // std::cout<<"rngX:"<<rngX<<" rngY:"<<rngY<<"\n";
      e = get_Edge_indices_PKSG(offX, rngX, offY, rngY,  std::ref(distribution), std::ref(generator), a, b, c, d, u, k);
      h_idx = e.first;
      v_idx = e.second;
      // h_idx = genEdgeIndex_FP(squ.get_X_start(), squ.get_X_end(), RMAT_a, RMAT_c, std::ref(dis), std::ref(gen));
      // v_idx = genEdgeIndex_FP(squ.get_Y_start(), squ.get_Y_end(), RMAT_a, RMAT_b, std::ref(dis), std::ref(gen));
      if( (!applyCondition && h_idx > v_idx) || (!allowEdgeToSelf && h_idx == v_idx ) ) // Short-circuit if it doesn't pass the test.
        continue;
      if( createNewEdges )  // Create new edges.
        edgesVec.push_back( Edge( h_idx, v_idx ) );
      else  // Replace non-valids.
        edgesVec[duplicate_indices[numEdges]] = ( Edge( h_idx, v_idx ) );
      ++edgeIdx;
      ++numEdges;

    }
  }
  // std::cout<<"EXITED\n";
  // Generate X ∼ Poisson(Ep)
  // //For each edge determine destination vertex
  // for i = 1, · · · X do
  // v = 0; z = u
  // for j = 1, · · · k do
  // l = mod(z, n)
  // With probability Vls choose subregion
  // s
  // v = nv + s; z = z/n (integer division)
  // end for
  // Add edge (u, v)
  // end for
  // end for


}

void generate_edges( Square& squ,
		std::vector<Edge>& edgesVec,
		const double RMAT_a, const double RMAT_b, const double RMAT_c,
		const bool directedGraph,
		const bool allowEdgeToSelf,
		std::uniform_int_distribution<>& dis, std::mt19937_64& gen,
		std::vector<unsigned long long>& duplicate_indices ) {

	auto applyCondition = directedGraph || ( squ.get_H_idx() < squ.get_V_idx() ); // true: if the graph is directed or in case it is undirected, the square belongs to the lower triangle of adjacency matrix. false: the diagonal passes the rectangle and the graph is undirected.
	auto createNewEdges = duplicate_indices.empty();
	unsigned long long nEdgesToGen = createNewEdges ? squ.getnEdges() : duplicate_indices.size();
  static std::default_random_engine generator;

  std::uniform_real_distribution<double> distribution(0.0,1.0);
  double sumA[128], sumAB[128], sumABC[128], sumAC[128];
  for (int i = 0; i < 128; ++i)
  {
    double A = RMAT_a * (distribution(generator)+0.5);
    double B = RMAT_b * (distribution(generator)+0.5);
    double C = RMAT_c *(distribution(generator)+0.5);
    double D = (1- RMAT_a-RMAT_b-RMAT_c) *(distribution(generator)+0.5);
    double abcd = A+B+C+D;
    sumA[i] = A/(abcd);
    sumAB[i] = (A+B)/abcd;
    sumAC[i] = (A+C)/abcd;
    sumABC[i] = (A+B+C)/abcd;
  }
        
	for( unsigned long long edgeIdx = 0; edgeIdx < nEdgesToGen; ) {
		unsigned long long h_idx, v_idx;
		std::pair <long long int,long long int> e;
		unsigned long long offX, offY, rngX, rngY;
		offX = squ.get_X_start();
		rngX = squ.get_X_end()-offX;
		offY = squ.get_Y_start();
		rngY = squ.get_Y_end()-offY;
    // std::cout<<"rngX:"<<rngX<<" rngY:"<<rngY<<"\n";
		e = get_Edge_indices(offX, rngX, offY, rngY,RMAT_a,RMAT_b, RMAT_c,  std::ref(distribution), std::ref(generator), sumA, sumAB, sumAC, sumABC );
		h_idx = e.first;
		v_idx = e.second;
		// h_idx = genEdgeIndex_FP(squ.get_X_start(), squ.get_X_end(), RMAT_a, RMAT_c, std::ref(dis), std::ref(gen));
		// v_idx = genEdgeIndex_FP(squ.get_Y_start(), squ.get_Y_end(), RMAT_a, RMAT_b, std::ref(dis), std::ref(gen));
		if( (!applyCondition && h_idx > v_idx) || (!allowEdgeToSelf && h_idx == v_idx ) ) // Short-circuit if it doesn't pass the test.
			continue;
		if( createNewEdges )	// Create new edges.
			edgesVec.push_back(	Edge( h_idx, v_idx ) );
		else	// Replace non-valids.
			edgesVec[duplicate_indices[edgeIdx]] = ( Edge( h_idx, v_idx ) );
		++edgeIdx;
	}

}

void progressBar() {
	if( SHOW_PROGRESS_BARS ) {
		std::cout <<'|';
		std::cout.flush();
	}
}

void progressBarNewLine() {
	if( SHOW_PROGRESS_BARS )
		std::cout << std::endl;
}
