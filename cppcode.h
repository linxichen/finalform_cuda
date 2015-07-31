// This header should only contain routines that can only be compiled by
// gcc48, for example things utilize std::random, armadillo, and LAPACK

void mynormalcpp(double *, double, double, int, unsigned);
void myuniformcpp(double *, int, unsigned);
void myexponentialcpp(double *, int, unsigned);
void mytest(const int);
void fromchebydomain(double, double, int, double*);
void ind2sub(int, int*, int, int*);
void findprojector(double*, int, int, double*);

void linspace(double, double, int, double*);
void chebyspace(double, double, int, double*);
typedef void (*gridgen_fptr)(double, double, int, double*);
void tauchen_vec(int, int, int, double*, double*, double*, double*, gridgen_fptr);

// void qzdecomp(arma::mat &, arma::mat &, arma::mat &, arma::mat &);
void test();
void linearQZ(double*, double*, double*, double*, int, int, int, double*);
