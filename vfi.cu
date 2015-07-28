#define nk 100
#define nx 7
#define nz 7
#define nssigmax 2
#define ns nx*nz*nssigmax
#define nK 25
#define nq 25
#define nmarkup 15
#define tauchenwidth 2.5
#define tol 1e-3
#define outertol 1e-3
#define damp 0.5
#define maxiter 2000
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

// finds operating profit y-wl at each state given agg rules
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

// finds operating profit y-wl at each state given agg rules
struct updateU
{
	// Data member
	double *profit, *k_grid, *K_grid, *x_grid, *z_grid, *ssigmax_grid;
	double *q_grid, *P;
	double *U, *V;
	para p;
	aggrules r;

	// Construct this object, create util from _util, etc.
	__host__ __device__
	updateU(
		double* profit_ptr,
		double* k_grid_ptr,
		double* K_grid_ptr,
		double* x_grid_ptr,
		double* z_grid_ptr,
		double* ssigmax_grid_ptr,
		double* q_grid_ptr,
		double* P_ptr,
		double* U_ptr,
		double* V_ptr,
		para _p,
		aggrules _r
	) {
		profit       = profit_ptr;
		k_grid       = k_grid_ptr;
		K_grid       = K_grid_ptr;
		x_grid       = x_grid_ptr;
		z_grid       = z_grid_ptr;
		ssigmax_grid = ssigmax_grid_ptr;
		q_grid       = q_grid_ptr;
		P = P_ptr,
		U            = U_ptr,
		V            = V_ptr,
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
		int i_z       = subs[1];
		int i_ssigmax = subs[2];
		double z       = z_grid[i_z];
		double ssigmax = ssigmax_grid[i_ssigmax];
		double C       = exp( r.pphi_CC + r.pphi_CK*log(K) + r.pphi_Cz*log(z) + r.pphi_Cssigmax*log(ssigmax)  );
		double Kplus   = exp( r.pphi_KC + r.pphi_KK*log(K) + r.pphi_Kz*log(z) + r.pphi_Kssigmax*log(ssigmax)  );
		double qplus   = exp( r.pphi_qC + r.pphi_qK*log(K) + r.pphi_qz*log(z) + r.pphi_qssigmax*log(ssigmax)  );
		double llambda = 1/C;

		// find the indexes of (1-ddelta)*k
		int noinvest_ind = fit2grid((1-p.ddelta)*k,nk,k_grid);
		int i_left, i_right;
		if (noinvest_ind==nk-1) { // (1-ddelta)k>=maxK, then should use K[nk-2] as left point to extrapolate
			i_left = nk-2;
			i_right = nk-1;
		} else {
			i_left = noinvest_ind;
			i_right = noinvest_ind+1;
		};
		double kplus_left  = k_grid[i_left];
		double kplus_right = k_grid[i_right];
		int i_Kplus = fit2grid(Kplus,nK,K_grid);
		int i_qplus = fit2grid(qplus,nq,q_grid);

		// find EV_noinvest
		double EV_noinvest = 0;
		for (int i_splus = 0; i_splus < ns; i_splus++) {
			EV_noinvest += P[i_s+i_splus*ns]*linear_interp( (1-p.ddelta)*k, kplus_left, kplus_right, V[i_left+i_splus*nk+i_Kplus*nk*ns+i_qplus*nk*ns*nK], V[i_right+i_splus*nk+i_Kplus*nk*ns+i_qplus*nk*ns*nK]);
		};

		// Find U finally
		U[index] = llambda*profit[index] + p.bbeta*EV_noinvest;
	};
};

// finds W, and thus V because U is assumed to be computed beforehand!!
struct updateWV
{
	// Data member
	double *profit, *k_grid, *K_grid, *x_grid, *z_grid, *ssigmax_grid;
	double *q_grid, *P;
	double *W, *U, *V;
	double *Vplus, *kopt;
	int    *active, *koptind;
	para p;
	aggrules r;

	// Construct this object, create util from _util, etc.
	__host__ __device__
	updateWV(
		double* profit_ptr,
		double* k_grid_ptr,
		double* K_grid_ptr,
		double* x_grid_ptr,
		double* z_grid_ptr,
		double* ssigmax_grid_ptr,
		double* q_grid_ptr,
		double* P_ptr,
		double* W_ptr,
		double* U_ptr,
		double* V_ptr,
		double* Vplus_ptr,
		double* kopt_ptr,
		int*    active_ptr,
		int*    koptind_ptr,
		para _p,
		aggrules _r
	) {
		profit       = profit_ptr;
		k_grid       = k_grid_ptr;
		K_grid       = K_grid_ptr;
		x_grid       = x_grid_ptr;
		z_grid       = z_grid_ptr;
		ssigmax_grid = ssigmax_grid_ptr;
		q_grid       = q_grid_ptr;
		P            = P_ptr,
		W            = U_ptr,
		U            = U_ptr,
		V            = V_ptr,
		Vplus        = Vplus_ptr,
		p            = _p;
		r            = _r;
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

		// Find aggregate stuff
		double k = k_grid[i_k];
		double K = K_grid[i_K];
		double q = q_grid[i_q];
		int subs_shock [3];
		int size_vec_shock [3];
		size_vec_shock[0] = nx;
		size_vec_shock[1] = nz;
		size_vec_shock[2] = nssigmax;
		ind2sub(3,size_vec_shock,i_s,subs_shock);
		int i_z           = subs_shock[1];
		int i_ssigmax     = subs_shock[2];
		double z          = z_grid[i_z];
		double ssigmax    = ssigmax_grid[i_ssigmax];
		double C      = exp(r.pphi_CC      + r.pphi_CK*log(K)      + r.pphi_Cz*log(z)      + r.pphi_Cssigmax*log(ssigmax)      );
		double Kplus  = exp(r.pphi_KC      + r.pphi_KK*log(K)      + r.pphi_Kz*log(z)      + r.pphi_Kssigmax*log(ssigmax)      );
		double qplus  = exp(r.pphi_qC      + r.pphi_qK*log(K)      + r.pphi_qz*log(z)      + r.pphi_qssigmax*log(ssigmax)      );
		double ttheta = exp(r.pphi_tthetaC + r.pphi_tthetaK*log(K) + r.pphi_tthetaz*log(z) + r.pphi_tthetassigmax*log(ssigmax) + r.pphi_tthetaq*log(q) );
		double llambda = 1/C;
		double mmu = p.aalpha*pow(ttheta,p.aalpha1);

		// find the indexes of (1-ddelta)*k
		int noinvest_ind = fit2grid((1-p.ddelta)*k,nk,k_grid);
		int i_left_noinv, i_right_noinv;
		if (noinvest_ind==nk-1) { // (1-ddelta)k>=maxK, then should use K[nk-2] as left point to extrapolate
			i_left_noinv = nk-2;
			i_right_noinv = nk-1;
		} else {
			i_left_noinv = noinvest_ind;
			i_right_noinv = noinvest_ind+1;
		};
		double kplus_left_noinv  = k_grid[i_left_noinv];
		double kplus_right_noinv = k_grid[i_right_noinv];
		int i_Kplus = fit2grid(Kplus,nK,K_grid);
		int i_qplus = fit2grid(qplus,nq,q_grid);

		// find EV_noinvest
		double EV_noinvest = 0;
		for (int i_splus = 0; i_splus < ns; i_splus++) {
			EV_noinvest += P[i_s+i_splus*ns]*linear_interp( (1-p.ddelta)*k, kplus_left_noinv, kplus_right_noinv, V[i_left_noinv+i_splus*nk+i_Kplus*nk*ns+i_qplus*nk*ns*nK], V[i_right_noinv+i_splus*nk+i_Kplus*nk*ns+i_qplus*nk*ns*nK]);
		};

		// search through all positve investment level
		double rhsmax = -999999999999999;
		int koptind_active = 0;
		for (int i_kplus = 0; i_kplus < nk; i_kplus++) {
			double convexadj = p.eeta*k*pow((k_grid[i_kplus]-(1-p.ddelta)*k)/k,2);
			double effective_price = (k_grid[i_kplus]>(1-p.ddelta)*k) ? q : p.pphi*q;
			// compute kinda stupidly EV
			double EV = 0;
			for (int i_splus = 0; i_splus < ns; i_splus++) {
				EV += P[i_s+i_splus*ns]*V[i_kplus+i_splus*nk+i_Kplus*nk*ns+i_qplus*nk*ns*nK];
			};

			double candidate = llambda*profit[i_k+i_s*nk+i_K*nk*ns] + mmu*( llambda*(-effective_price)*(k_grid[i_kplus]-(1-p.ddelta)*k) - llambda*convexadj + p.bbeta*EV ) + (1-mmu)*EV_noinvest;
			if (candidate > rhsmax) {
				rhsmax         = candidate;
				koptind_active = i_kplus;
			};
		};

		// Find U finally
		W[index] = rhsmax;
		if (W[index] > U[i_k+i_s*nk+i_K*nk*ns]) {
			Vplus[index]   = W[index];
			active[index]  = 1;
			koptind[index] = koptind_active;
			kopt[index]    = k_grid[koptind_active];
		} else {
			Vplus[index] = U[i_k+i_s*nk+i_K*nk*ns];
			active[index]  = 1;
			koptind[index] = noinvest_ind;
			kopt[index]    = (1-p.ddelta)*k;
		};
	};
};
// This unctor calculates the distance
struct myDist {
	// Tple is (V1low,Vplus1low,V1high,Vplus1high,...)
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
	p.eeta         = 0.1;
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
	h_vec_d h_U(nk*ns*nK,0.0);
	h_vec_d h_EV(nk*ns*nK*nq,0.0);
	h_vec_d h_profit(nk*ns*nK,0.0);
	h_vec_i h_koptind(nk*ns*nK*nq,0.0);
	h_vec_d h_kopt(nk*ns*nK*nq,0.0);
	h_vec_i h_active(nk*ns*nK*nq,0.0);

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
	d_vec_d d_q_grid       = h_q_grid;
	d_vec_d d_ssigmax_grid = h_ssigmax_grid;
	d_vec_d d_profit       = h_profit;
	d_vec_d d_V            = h_V;
	d_vec_d d_Vplus        = h_Vplus;
	d_vec_d d_W            = h_W;
	d_vec_d d_U            = h_U;
	d_vec_d d_EV           = h_EV;
	d_vec_d d_P            = h_P;
	d_vec_d d_kopt         = h_kopt;
	d_vec_i d_koptind      = h_koptind;
	d_vec_i d_active       = h_active;

	// Obtain device pointers to be used by cuBLAS
	double* d_k_grid_ptr       = raw_pointer_cast(d_k_grid.data());
	double* d_K_grid_ptr       = raw_pointer_cast(d_K_grid.data());
	double* d_x_grid_ptr       = raw_pointer_cast(d_x_grid.data());
	double* d_z_grid_ptr       = raw_pointer_cast(d_z_grid.data());
	double* d_ssigmax_grid_ptr = raw_pointer_cast(d_ssigmax_grid.data());
	double* d_profit_ptr       = raw_pointer_cast(d_profit.data());
	double* d_q_grid_ptr       = raw_pointer_cast(d_q_grid.data());
	double* d_V_ptr            = raw_pointer_cast(d_V.data());
	double* d_W_ptr            = raw_pointer_cast(d_W.data());
	double* d_Vplus_ptr        = raw_pointer_cast(d_Vplus.data());
	double* d_U_ptr            = raw_pointer_cast(d_U.data());
	double* d_P_ptr            = raw_pointer_cast(d_P.data());
	double* d_kopt_ptr         = raw_pointer_cast(d_kopt.data());
	int* d_koptind_ptr         = raw_pointer_cast(d_koptind.data());
	int* d_active_ptr          = raw_pointer_cast(d_active.data());

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
		// find profit at (i_k,i_s,i_K)
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

		// find U currently
		thrust::for_each(
			begin_noq,
			end_noq,
			updateU(
				d_profit_ptr,
				d_k_grid_ptr,
				d_K_grid_ptr,
				d_x_grid_ptr,
				d_z_grid_ptr,
				d_ssigmax_grid_ptr,
				d_q_grid_ptr,
				d_P_ptr,
				d_U_ptr,
				d_V_ptr,
				p,
				r
			)
		);

		// find U currently
		thrust::for_each(
			begin,
			end,
			updateWV(
				d_profit_ptr,
				d_k_grid_ptr,
				d_K_grid_ptr,
				d_x_grid_ptr,
				d_z_grid_ptr,
				d_ssigmax_grid_ptr,
				d_q_grid_ptr,
				d_P_ptr,
				d_W_ptr,
				d_U_ptr,
				d_V_ptr,
				d_Vplus_ptr,
				d_kopt_ptr,
				d_active_ptr,
				d_koptind_ptr,
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
	h_koptind = d_koptind;
	h_active  = d_active;
	h_profit  = d_profit;

    save_vec(h_K_grid,"./results/K_grid.csv");   // in #include "cuda_helpers.h"
    save_vec(h_k_grid,"./results/k_grid.csv");   // in #include "cuda_helpers.h"
	save_vec(h_V,"./results/Vgrid.csv");         // in #include "cuda_helpers.h"
	save_vec(h_active,"./results/active.csv");   // in #include "cuda_helpers.h"
	save_vec(h_koptind,"./results/koptind.csv"); // in #include "cuda_helpers.h"
	save_vec(h_kopt,"./results/kopt.csv");       // in #include "cuda_helpers.h"
	std::cout << "Policy functions output completed." << std::endl;

	// Export parameters to MATLAB
	p.exportmatlab("./MATLAB/vfi_para.m");

	return 0;
}
