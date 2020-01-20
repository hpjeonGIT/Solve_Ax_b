#include <algorithm>
#include "dataset.h"

void Dataset::random_matrix(const int &n){
	A_.resize(n*n);
	b_.resize(n);
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0,1);
	for (int i=0;i<n;i++){
	    for (int j=0;j<n;j++){
		A_[j*n+i] = distribution(generator);
	    }
	    b_[i] = distribution(generator);
	}
}

void Dataset::sym_matrix(const int &n) {
	A_.resize(n*n);
	b_.resize(n);
	for (int i=0;i<n;i++){
	    for (int j=0;j<n;j++){
		A_[j*n+i] = (double) (n - abs(i-j));
	    }
	    b_[i] = (double) abs(n - 2*i);
	}
}

std::vector<double> Dataset::return_A() {
	return A_;
}

std::vector<double> Dataset::return_b() {
	return b_;
}
