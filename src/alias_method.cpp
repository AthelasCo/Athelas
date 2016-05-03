// File: alias_method.cpp
// Author: Jaewon Jung (jj@notsharingmy.info)
//
// This is a C++ port of Keith's Java implementation here:
//  http://w...content-available-to-author-only...z.com/interesting/code/?dir=alias-method
//
// An implementation of the alias method implemented using Vose's algorithm.
// The alias method allows for efficient sampling of random values from a
// discrete probability distribution (i.e. rolling a loaded die) in O(1) time
// each after O(n) preprocessing time.
//
// For a complete writeup on the alias method, including the intuition and
// important proofs, please see the article "Darts, Dice, and Coins: Smpling
// from a Discrete Distribution" at
//
//                 http://w...content-available-to-author-only...z.com/darts-dice-coins/
//
#include <random>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <ctime>

// Some utility class and function for wrapping the C rand() library function 
// struct rand_engine
// {
//     int operator() ()
//     {
//         return rand();
//     }
// };

template <class T>
double gen_uniform_normalized_double(T& random)
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(random);
}

// template <>
// double gen_uniform_normalized_double<rand_engine>(rand_engine& random)
// {
//     return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
// }

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

int main(int argc, char* argv[])
{
    std::vector<double> probabilities;
    probabilities.push_back(1.0/20.0);
    probabilities.push_back(7.0/20.0);
    probabilities.push_back(3.0/20.0);
    probabilities.push_back(5.0/20.0);
    probabilities.push_back(4.0/20.0);

    //rand_engine rnd;
    //srand(clock());
    // std::mt19937 rnd;
    std::mt19937_64 rnd;
    rnd.seed(clock());
    auto am = make_alias_method(probabilities, rnd);

    const int tries = 1000000;
    int x;
    int count[5] = { 0, 0, 0, 0, 0 };
    for(int i=0; i<tries; ++i)
    {
        x = am.next();
        ++count[x];
        printf("%d \n", x);
    }

    for(int i=0; i<5; ++i)
        printf("%f\n", 20.0*static_cast<double>(count[i])/static_cast<double>(tries));

    return 0;
}