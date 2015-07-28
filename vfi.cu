#define nk 250
#define nx 7
#define nz 7
#define nssigmax 2
#define ns nx*nz*nssigmax
#define nK 25
#define nq 25
#define nmarkup 25
#define tauchenwidth 2.5
#define tol 1e-3
#define outertol 1e-3
#define damp 0.5
#define maxiter 200
#define SIMULPERIOD 3000
#define kwidth 1.5

/* Includes, system */
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

// Includes, Thrust
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

// Includes, cuda
#include <cublas_v2.h>
#include "cuda_helpers.h"

// Includes, my own creation
#include "common.h"

// Includes model stuff
#include "invpricemodel.h"

/// This function finds the value of RHS given k', k, z
__host__ __device__
double rhsvalue (state s, int i_z, double kplus, int i_kplus, double* EV, para p) {
	return 0;
};

// This functor optimal kplus and Vplus
struct updateprofit
{
	// Data member
	double *profit, *k_grid, *K_grid, *x_grid, *z_grid, *ssigmax_grid;
	para p;
	aggrules r;

	// Construct this object, create util from _util, etc.
	__host__ __device__
	updateprofit(
		double* profit_ptr,
		double* k_grid_ptr,
		double* K_grid_ptr,
		double* x_grid_ptr,
		double* z_grid_ptr,
		double* ssigmax_grid_ptr,
		para _p,
		aggrules _r
	) {
		profit       = profit_ptr;
		k_grid       = k_grid_ptr;
		K_grid       = K_grid_ptr;
		x_grid       = x_grid_ptr;
		z_grid       = z_grid_ptr;
		ssigmax_grid = ssigmax_grid_ptr;
		p            = _p;
		r            = _r;
	};

	__host__ __device__
	void operator()(int index) {
		// Perform ind2sub
		int subs[3];
		int size_vec[3];
		size_vec[0] = nk;
		size_vec[1] = ns;
		size_vec[2] = nK;
		ind2sub(3,size_vec,index,subs);
		int i_k = subs[0];
		int i_s = subs[1];
		int i_K = subs[2];

		// Find aggregate stuff
		double k = k_grid[i_k];
		double K = K_grid[i_K];
		size_vec[0] = nx;
		size_vec[1] = nz;
		size_vec[2] = nssigmax;
		ind2sub(3,size_vec,i_s,subs);
		int i_x       = subs[0];
		int i_z       = subs[1];
		int i_ssigmax = subs[2];
		double x       = x_grid[i_x];
		double z       = z_grid[i_z];
		double ssigmax = ssigmax_grid[i_ssigmax];
		double C = exp( r.pphi_CC + r.pphi_CK*log(K) + r.pphi_Cz*log(z) + r.pphi_Cssigmax*log(ssigmax)  );
		double w = p.ppsi_n*C;

		// Find profit finally
		double l = pow( w/z/x/p.v/pow(k,p.aalpha), 1.0/(p.v-1) );
		profit[index] = z*x*pow(k,p.aalpha)*pow(l,p.v) - w*l;
	};
};
/*
// This functor optimal kplus and Vplus
struct updateW
{
	// Data member
	double *k_grid, *K_grid, *Z;
	int *koptind;
    double *Vplus;
	para p;
	aggrules r;

	// Construct this object, create util from _util, etc.
	__host__ __device__
    updateW(double* K_ptr, double* Z_ptr, double* EV_ptr, int* koptind_ptr, double* Vplus_ptr, para _p) {
		K = K_ptr; Z = Z_ptr; EV = EV_ptr;
		koptind = koptind_ptr; Vplus = Vplus_ptr;
		p = _p;
	};

	__host__ __device__
	void operator()(int index) {
		// Perform ind2sub
		int subs[4];
		int size_vec[4];
		size_vec[0] = nk;
		size_vec[1] = ns;
		size_vec[2] = nK;
		size_vec[3] = nq;
		ind2sub(4,size_vec,index,subs);
		int i_k = subs[0];
		int i_s = subs[1];
		int i_K = subs[2];
		int i_q = subs[3];

		// Find and construct state and control, otherwise they won't update in the for loop
		double k =K[i_k]; double z=Z[i_z];

		// Exploit concavity to update V

	};
};
*/

// This functor calculates the distance
struct myDist {
	// Tuple is (V1low,Vplus1low,V1high,Vplus1high,...)
	template <typename Tuple>
		__host__ __device__
	double operator()(Tuple t)
	{
		return abs(thrust::get<0>(t)-thrust::get<1>(t));
	}
};

int main(int argc, char ** argv)
{
	// Select Device from the first argument of main
	int num_devices;
	cudaGetDeviceCount(&num_devices);
	if (argc > 1) {
		int gpu = min(num_devices,atoi(argv[1]));
		cudaSetDevice(gpu);
	};

	// set parameters
	para p; // in #include "invpricemodel.h"
	p.bbeta        = 0.99;
	p.ttau         = 0.1;
	p.aalpha       = 0.25;
	p.v            = 0.5;
	p.ddelta       = .1/double(4);
	p.pphi         = 0.000000;
	p.MC           = 1;
	p.rrhox        = 0.95;
	p.ppsi         = 0.00;
	p.rrhoz        = p.rrhox;
	p.ssigmaz      = 0.01;
	p.ssigmax_low  = 0.04;
	p.ssigmax_high = 0.04*3;
	p.ppsi_n       = 1;
	p.aalpha0      = 0.95;
	p.aalpha1      = 0.01;
	p.Pssigmax[0] = 0.95; p.Pssigmax[2] = 0.05;
	p.Pssigmax[1] = 0.08; p.Pssigmax[3] = 0.92;

	// Create all STATE, SHOCK grids here
	h_vec_d h_k_grid(nk);
	h_vec_d h_K_grid(nK);
	h_vec_d h_z_grid(nz);
    h_vec_d h_x_grid(nx);
    h_vec_d h_ssigmax_grid(nssigmax);
    h_vec_d h_q_grid(nq);
    h_vec_d h_markup_grid(nmarkup);
    h_vec_d h_logZ(nz);
    h_vec_d h_logX(nx);
	h_vec_d h_PZ(nz*nz, 0.0);
	h_vec_d h_PX_low(nx*nx, 0.0);
	h_vec_d h_PX_high(nx*nx, 0.0);
	h_vec_d h_P(ns*ns, 0.0);
	h_vec_d h_V(nk*ns*nK*nq,0.0);
	h_vec_d h_Vplus(nk*ns*nK*nq,0);
	h_vec_d h_W(nk*ns*nK*nq,0.0);
	h_vec_d h_Wplus(nk*ns*nK*nq,0.0);
	h_vec_d h_U(nk*ns*nK,0.0);
	h_vec_d h_Uplus(nk*ns*nK,0.0);
	h_vec_i h_koptind(nk*ns*nK*nq,0.0);
	h_vec_d h_EV(nk*ns*nK*nq,0.0);
	h_vec_d h_profit(nk*ns*nK,0.0);

	// load_vec(h_V,"./results/Vgrid.csv"); // in #include "cuda_helpers.h"

	// Create capital grid
	double minK = 0.5;
	double maxK = 20.0;
	linspace(minK,maxK,nk,thrust::raw_pointer_cast(h_k_grid.data())); // in #include "cuda_helpers.h"
	linspace(minK,maxK,nK,thrust::raw_pointer_cast(h_K_grid.data())); // in #include "cuda_helpers.h"

	// Create shocks grids
	h_ssigmax_grid[0] = p.ssigmax_low;
	h_ssigmax_grid[1] = p.ssigmax_high;
	double* h_logZ_ptr = thrust::raw_pointer_cast(h_logZ.data());
	double* h_PZ_ptr   = thrust::raw_pointer_cast(  h_PZ.data());
	tauchen(p.rrhoz, p.ssigmaz, h_logZ, h_PZ, tauchenwidth); // in #include "cuda_helpers.h"
	for (int i_z = 0; i_z < nz; i_z++) {
		h_z_grid[i_z] = exp(h_logZ[i_z]);
	};
	double* h_logX_ptr    = thrust::raw_pointer_cast(h_logX   .data());
	double* h_PX_low_ptr  = thrust::raw_pointer_cast(h_PX_low .data());
	double* h_PX_high_ptr = thrust::raw_pointer_cast(h_PX_high.data());
	tauchen(           p.rrhox, p.ssigmax_high, h_logX, h_PX_high, tauchenwidth); // in #include "cuda_helpers.h"
	tauchen_givengrid( p.rrhox, p.ssigmax_low,  h_logX, h_PX_low,  tauchenwidth); // in #include "cuda_helpers.h"
	for (int i_x = 0; i_x < nx; i_x++) {
		h_x_grid[i_x] = exp(h_logX[i_x]);
	};
	// find combined transition matrix P
	for (int i_s = 0; i_s < ns; i_s++) {
		int subs[3];
		int size_vec[3];
		size_vec[0] = nx;
		size_vec[1] = nz;
		size_vec[2] = nssigmax;
		ind2sub(3,size_vec,i_s,subs);
		int i_x       = subs[0];
		int i_z       = subs[1];
		int i_ssigmax = subs[2];
		for (int i_splus = 0; i_splus < ns; i_splus++) {
			int subs[3];
			int size_vec[3];
			size_vec[0] = nx;
			size_vec[1] = nz;
			size_vec[2] = nssigmax;
			ind2sub(3,size_vec,i_splus,subs);
			int i_xplus       = subs[0];
			int i_zplus       = subs[1];
			int i_ssigmaxplus = subs[2];
			if (i_ssigmaxplus==0) {
				h_P[i_s+i_splus*ns] = h_PX_low[i_x+i_xplus*nx]*h_PZ[i_z+i_zplus*nz]* p.Pssigmax[i_ssigmax+i_ssigmaxplus*nssigmax];
			} else {
				h_P[i_s+i_splus*ns] = h_PX_high[i_x+i_xplus*nx]*h_PZ[i_z+i_zplus*nz]*p.Pssigmax[i_ssigmax+i_ssigmaxplus*nssigmax];
			}
		};
	};
	display_vec(h_P);

	// Create pricing grids
	double minq = 0.4;
	double maxq = 2.0;
	double minmarkup = 1.0;
	double maxmarkup = 1.3;
	linspace(minq,maxq,nq,thrust::raw_pointer_cast(h_q_grid.data())); // in #include "cuda_helpers.h"
	linspace(minmarkup,maxmarkup,nmarkup,thrust::raw_pointer_cast(h_markup_grid.data())); // in #include "cuda_helpers.h"

	// set initial agg rules
	aggrules r;
	r.pphi_qC = log((maxq+minq)/2.0); // constant term
	r.pphi_qK = 0; // w.r.t agg K
	r.pphi_qz = 0; // w.r.t agg TFP
	r.pphi_qssigmax = 0; // w.r.t uncertainty
	r.pphi_KK = 0.99;
	r.pphi_KC = log((maxK+minK)/2.0);
	r.pphi_Kz = 0;
	r.pphi_Kssigmax = 0;// Aggregate Law of motion for aggregate capital
	r.pphi_CC = log((maxq+minq)/(maxmarkup+minmarkup)/p.ppsi_n);
	r.pphi_CK = 0.0;
	r.pphi_Cz = 0.0;
	r.pphi_Cssigmax = 0.0;
	r.pphi_tthetaC = log(0.95); // tightness ratio depends on q
	r.pphi_tthetaK = 0.0;
	r.pphi_tthetaz = 0.0;
	r.pphi_tthetassigmax = 0.0;
	r.pphi_tthetaq = 0.1;// lower q -- more firms invest -- lower ttheta

	// Copy to the device
	d_vec_d d_k_grid       = h_k_grid;
	d_vec_d d_K_grid       = h_K_grid;
	d_vec_d d_x_grid       = h_x_grid;
	d_vec_d d_z_grid       = h_z_grid;
	d_vec_d d_ssigmax_grid = h_ssigmax_grid;
	d_vec_d d_profit       = h_profit;
	d_vec_d d_V            = h_V;
	d_vec_d d_Vplus        = h_Vplus;
	d_vec_i d_koptind      = h_koptind;
	d_vec_d d_EV           = h_EV;
	d_vec_d d_P            = h_P;

	// Obtain device pointers to be used by cuBLAS
	double* d_k_grid_ptr       = raw_pointer_cast(d_k_grid.data());
	double* d_K_grid_ptr       = raw_pointer_cast(d_K_grid.data());
	double* d_x_grid_ptr       = raw_pointer_cast(d_x_grid.data());
	double* d_z_grid_ptr       = raw_pointer_cast(d_z_grid.data());
	double* d_ssigmax_grid_ptr = raw_pointer_cast(d_ssigmax_grid.data());
	double* d_profit_ptr       = raw_pointer_cast(d_profit.data());
	int* d_koptind_ptr         = raw_pointer_cast(d_koptind.data());

	// Firstly a virtual index array from 0 to nk*nk*nz
	thrust::counting_iterator<int> begin(0);
	thrust::counting_iterator<int> end(nk*ns*nK*nq);
	thrust::counting_iterator<int> begin_noq(0);
	thrust::counting_iterator<int> end_noq(nk*ns*nK);

    // Create Timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    // Start Timer
	cudaEventRecord(start,NULL);

	double diff = 10;  int iter = 0;
	while ((diff>tol)&&(iter<maxiter)){
		// Directly find the new Value function
		thrust::for_each(
			begin_noq,
			end_noq,
			updateprofit(
				d_profit_ptr,
				d_k_grid_ptr,
				d_K_grid_ptr,
				d_x_grid_ptr,
				d_z_grid_ptr,
				d_ssigmax_grid_ptr,
				p,
				r
			)
		);

		// Find diff
		diff = thrust::transform_reduce(
			thrust::make_zip_iterator(thrust::make_tuple(d_V.begin(),d_Vplus.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_V.end()  ,d_Vplus.end())),
			myDist(),
			0.0,
			thrust::maximum<double>()
			);

// maximum in #include <thrust/extrema.h>

		std::cout << "diff is: "<< diff << std::endl;

		// update correspondence
		d_V = d_Vplus;

		std::cout << ++iter << std::endl;
		std::cout << "=====================" << std::endl;

	};

	// Stop Timer
	cudaEventRecord(stop,NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0;
	cudaEventElapsedTime(&msecTotal, start, stop);

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal;
	std::cout << "Time= " << msecPerMatrixMul/1000 << " secs, iter= " << iter << std::endl;

	// Copy back to host and print to file
	h_V       = d_V;
	h_EV      = d_EV;
	h_koptind = d_koptind;
	h_profit  = d_profit;
	display_vec(h_profit);

	/*
    save_vec(h_K,"./results/Kgrid.csv"); // in #include "cuda_helpers.h"
	save_vec(h_Z,"./results/Zgrid.csv"); // in #include "cuda_helpers.h"
	save_vec(h_P,"./results/Pgrid.csv"); // in #include "cuda_helpers.h"
	save_vec(h_V,"./results/Vgrid.csv"); // in #include "cuda_helpers.h"
	std::cout << "Policy functions output completed." << std::endl;
	*/

	// Export parameters to MATLAB
	p.exportmatlab("./MATLAB/vfi_para.m");

	return 0;
}
