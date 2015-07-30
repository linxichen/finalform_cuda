#define nk 300
#define nx 7
#define nz 7
#define nssigmax 2
#define ns nx*nz*nssigmax
#define nK 25
#define nq 25
#define nmarkup 15
#define tauchenwidth 2.5
#define tol 1e-2
#define outertol 1e-2
#define damp 0.5
#define maxiter 2000
#define SIMULPERIOD 1000
#define nhousehold 10000
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
#include <curand.h>

// Includes, my own creation
#include "common.h"

// Includes model stuff
#include "invpricemodel.h"

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
		int i_s = index/(nk*nK);
		int i_K = (index-i_s*nk*nK)/(nk);
		int i_k = (index-i_s*nk*nK-i_K*nk)/(1);

		// Find aggregate stuff
		double k = k_grid[i_k];
		double K = K_grid[i_K];
		int i_ssigmax = i_s/(nx*nz);
		int i_z       = (i_s-i_ssigmax*nx*nz)/(nx);
		int i_x       = (i_s-i_ssigmax*nx*nz-i_z*nx)/(1);
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
	double *q_grid, *EV;
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
		double* EV_ptr,
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
		EV           = EV_ptr,
		U            = U_ptr,
		V            = V_ptr,
		p            = _p;
		r            = _r;
	};

	__host__ __device__
	void operator()(int index) {
		// Perform ind2sub
		int i_s = index/(nk*nK);
		int i_K = (index-i_s*nk*nK)/(nk);
		int i_k = (index-i_s*nk*nK-i_K*nk)/(1);

		// Find aggregate stuff
		double k = k_grid[i_k];
		double K = K_grid[i_K];
		int i_ssigmax = i_s/(nx*nz);
		int i_z       = (i_s-i_ssigmax*nx*nz)/(nx);
		double z       = z_grid[i_z];
		double ssigmax = ssigmax_grid[i_ssigmax];
		double C       = exp( r.pphi_CC + r.pphi_CK*log(K) + r.pphi_Cz*log(z) + r.pphi_Cssigmax*log(ssigmax)  );
		double Kplus   = exp( r.pphi_KC + r.pphi_KK*log(K) + r.pphi_Kz*log(z) + r.pphi_Kssigmax*log(ssigmax)  );
		double qplus   = exp( r.pphi_qC + r.pphi_qK*log(K) + r.pphi_qz*log(z) + r.pphi_qssigmax*log(ssigmax)  );
		double llambda = 1/C;
		int i_Kplus = fit2grid(Kplus,nK,K_grid);
		int i_qplus = fit2grid(qplus,nq,q_grid);

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

		// find EV_noinvest
		double EV_noinvest = linear_interp( (1-p.ddelta)*k, kplus_left, kplus_right, EV[i_left+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq], EV[i_right+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq]);
		/* double EV_noinvest = EV[noinvest_ind+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq]; */

		// Find U finally
		U[index] = llambda*profit[index] + p.bbeta*EV_noinvest;
	};
};

// finds W, and thus V because U is assumed to be computed beforehand!!
struct updateWV
{
	// Data member
	double *profit, *k_grid, *K_grid, *x_grid, *z_grid, *ssigmax_grid;
	double *q_grid, *EV;
	double *W, *U, *V;
	double *Vplus, *kopt;
	int    *active, *koptindplus;
	para p;
	aggrules r;

	// Construct this object, create util from _util, etc.
	__host__ __device__
	updateWV(
		double*  profit_ptr,
		double*  k_grid_ptr,
		double*  K_grid_ptr,
		double*  x_grid_ptr,
		double*  z_grid_ptr,
		double*  ssigmax_grid_ptr,
		double*  q_grid_ptr,
		double*  EV_ptr,
		double*  W_ptr,
		double*  U_ptr,
		double*  V_ptr,
		double*  Vplus_ptr,
		double*  kopt_ptr,
		int*     active_ptr,
		int*     koptindplus_ptr,
		para     _p,
		aggrules _r
	) {
		profit       = profit_ptr;
		k_grid       = k_grid_ptr;
		K_grid       = K_grid_ptr;
		x_grid       = x_grid_ptr;
		z_grid       = z_grid_ptr;
		ssigmax_grid = ssigmax_grid_ptr;
		q_grid       = q_grid_ptr;
		EV           = EV_ptr,
		W            = U_ptr,
		U            = U_ptr,
		V            = V_ptr,
		Vplus        = Vplus_ptr,
		koptindplus  = koptindplus_ptr,
		kopt         = kopt_ptr,
		active       = active_ptr,
		p            = _p;
		r            = _r;
	};

	__host__ __device__
	void operator()(int index) {
		// Perform ind2sub
		int i_s = (index)/(nk*nK*nq);
		int i_q = (index-i_s*nk*nK*nq)/(nk*nK);
		int i_K = (index-i_s*nk*nK*nq-i_q*nk*nK)/(nk);
		int i_k = (index-i_s*nk*nK*nq-i_q*nk*nK-i_K*nk)/(1);

		// Find aggregate stuff
		int i_ssigmax = i_s/(nx*nz);
		int i_z       = (i_s-i_ssigmax*nx*nz)/(nx);
		double k       = k_grid[i_k];
		double K       = K_grid[i_K];
		double q       = q_grid[i_q];
		double z       = z_grid[i_z];
		double ssigmax = ssigmax_grid[i_ssigmax];
		double C      = exp(r.pphi_CC      + r.pphi_CK*log(K)      + r.pphi_Cz*log(z)      + r.pphi_Cssigmax*log(ssigmax)      );
		double Kplus  = exp(r.pphi_KC      + r.pphi_KK*log(K)      + r.pphi_Kz*log(z)      + r.pphi_Kssigmax*log(ssigmax)      );
		double qplus  = exp(r.pphi_qC      + r.pphi_qK*log(K)      + r.pphi_qz*log(z)      + r.pphi_qssigmax*log(ssigmax)      );
		double ttheta = exp(r.pphi_tthetaC + r.pphi_tthetaK*log(K) + r.pphi_tthetaz*log(z) + r.pphi_tthetassigmax*log(ssigmax) + r.pphi_tthetaq*log(q) );
		double llambda = 1/C;
		double mmu = p.aalpha*pow(ttheta,p.aalpha1);
		int i_Kplus = fit2grid(Kplus,nK,K_grid);
		int i_qplus = fit2grid(qplus,nq,q_grid);

		// find the indexes of (1-ddelta)*k
		int noinvest_ind = fit2grid((1-p.ddelta)*k,nk,k_grid);
		int i_left_noinv, i_right_noinv;
		if (noinvest_ind == nk-1) { // (1-ddelta)k>=maxK, then should use K[nk-2] as left point to extrapolate
			i_left_noinv = nk-2;
			i_right_noinv = nk-1;
		} else {
			i_left_noinv = noinvest_ind;
			i_right_noinv = noinvest_ind+1;
		};
		double kplus_left_noinv  = k_grid[i_left_noinv];
		double kplus_right_noinv = k_grid[i_right_noinv];

		// find EV_noinvest
		double EV_noinvest = linear_interp( (1-p.ddelta)*k, kplus_left_noinv, kplus_right_noinv, EV[i_left_noinv+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq], EV[i_right_noinv+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq]);
		/* double EV_noinvest = EV[noinvest_ind+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq]; */

		// search through all positve investment level
		double rhsmax = -999999999999999;
		int koptind_active = 0;
		for (int i_kplus = 0; i_kplus < nk; i_kplus++) {
			double convexadj = p.eeta*k*pow((k_grid[i_kplus]-(1-p.ddelta)*k)/k,2);
			double effective_price = (k_grid[i_kplus]>(1-p.ddelta)*k) ? q : p.pphi*q;
			// compute kinda stupidly EV
			double EV_inv = EV[i_kplus+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq];
			double candidate = llambda*profit[i_k+i_K*nk+i_s*nk*nK] + mmu*( llambda*(-effective_price)*(k_grid[i_kplus]-(1-p.ddelta)*k) - llambda*convexadj + p.bbeta*EV_inv ) + (1-mmu)*p.bbeta*EV_noinvest;
			if (candidate > rhsmax) {
				rhsmax         = candidate;
				koptind_active = i_kplus;
			};
		};

		// Find W and V finally
		Vplus[index] = rhsmax;
			koptindplus[index] = koptind_active;
		if (k_grid[koptind_active] != (1-p.ddelta)*k) {
			active[index]      = 1;
			kopt[index]        = k_grid[koptind_active];
		} else {
			active[index]      = 0;
			kopt[index]        = (1-p.ddelta)*k;
		};
	};
};

// simulate for each household
struct simulation
{
	// Data member
	double *profit, *k_grid, *K_grid, *x_grid, *z_grid, *ssigmax_grid;
	double *q_grid, *EV;
	double *W, *U, *V;
	double *Vplus, *kopt;
	int    *active, *koptindplus;
	para p;
	aggrules r;

	// Construct this object, create util from _util, etc.
	__host__ __device__
	simulation(
		double*  profit_ptr,
		double*  k_grid_ptr,
		double*  K_grid_ptr,
		double*  x_grid_ptr,
		double*  z_grid_ptr,
		double*  ssigmax_grid_ptr,
		double*  q_grid_ptr,
		double*  EV_ptr,
		double*  W_ptr,
		double*  U_ptr,
		double*  V_ptr,
		double*  Vplus_ptr,
		double*  kopt_ptr,
		int*     active_ptr,
		int*     koptindplus_ptr,
		para     _p,
		aggrules _r
	) {
		profit       = profit_ptr;
		k_grid       = k_grid_ptr;
		K_grid       = K_grid_ptr;
		x_grid       = x_grid_ptr;
		z_grid       = z_grid_ptr;
		ssigmax_grid = ssigmax_grid_ptr;
		q_grid       = q_grid_ptr;
		EV           = EV_ptr,
		W            = U_ptr,
		U            = U_ptr,
		V            = V_ptr,
		Vplus        = Vplus_ptr,
		koptindplus  = koptindplus_ptr,
		kopt         = kopt_ptr,
		active       = active_ptr,
		p            = _p;
		r            = _r;
	};

	__host__ __device__
	void operator()(int index) {
		k_sim[index+0*nhousehold] = k_start;
		K_sim[index+0*nhousehold] = K_start;
		q_sim[index+0*nhousehold] = q_start;
		// Perform ind2sub
		int i_s = (index)/(nk*nK*nq);
		int i_q = (index-i_s*nk*nK*nq)/(nk*nK);
		int i_K = (index-i_s*nk*nK*nq-i_q*nk*nK)/(nk);
		int i_k = (index-i_s*nk*nK*nq-i_q*nk*nK-i_K*nk)/(1);

		// Find aggregate stuff
		int i_ssigmax = i_s/(nx*nz);
		int i_z       = (i_s-i_ssigmax*nx*nz)/(nx);
		double k       = k_grid[i_k];
		double K       = K_grid[i_K];
		double q       = q_grid[i_q];
		double z       = z_grid[i_z];
		double ssigmax = ssigmax_grid[i_ssigmax];
		double C      = exp(r.pphi_CC      + r.pphi_CK*log(K)      + r.pphi_Cz*log(z)      + r.pphi_Cssigmax*log(ssigmax)      );
		double Kplus  = exp(r.pphi_KC      + r.pphi_KK*log(K)      + r.pphi_Kz*log(z)      + r.pphi_Kssigmax*log(ssigmax)      );
		double qplus  = exp(r.pphi_qC      + r.pphi_qK*log(K)      + r.pphi_qz*log(z)      + r.pphi_qssigmax*log(ssigmax)      );
		double ttheta = exp(r.pphi_tthetaC + r.pphi_tthetaK*log(K) + r.pphi_tthetaz*log(z) + r.pphi_tthetassigmax*log(ssigmax) + r.pphi_tthetaq*log(q) );
		double llambda = 1/C;
		double mmu = p.aalpha*pow(ttheta,p.aalpha1);
		int i_Kplus = fit2grid(Kplus,nK,K_grid);
		int i_qplus = fit2grid(qplus,nq,q_grid);

		// find the indexes of (1-ddelta)*k
		int noinvest_ind = fit2grid((1-p.ddelta)*k,nk,k_grid);
		int i_left_noinv, i_right_noinv;
		if (noinvest_ind == nk-1) { // (1-ddelta)k>=maxK, then should use K[nk-2] as left point to extrapolate
			i_left_noinv = nk-2;
			i_right_noinv = nk-1;
		} else {
			i_left_noinv = noinvest_ind;
			i_right_noinv = noinvest_ind+1;
		};
		double kplus_left_noinv  = k_grid[i_left_noinv];
		double kplus_right_noinv = k_grid[i_right_noinv];

		// find EV_noinvest
		double EV_noinvest = linear_interp( (1-p.ddelta)*k, kplus_left_noinv, kplus_right_noinv, EV[i_left_noinv+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq], EV[i_right_noinv+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq]);
		/* double EV_noinvest = EV[noinvest_ind+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq]; */

		// search through all positve investment level
		double rhsmax = -999999999999999;
		int koptind_active = 0;
		for (int i_kplus = 0; i_kplus < nk; i_kplus++) {
			double convexadj = p.eeta*k*pow((k_grid[i_kplus]-(1-p.ddelta)*k)/k,2);
			double effective_price = (k_grid[i_kplus]>(1-p.ddelta)*k) ? q : p.pphi*q;
			// compute kinda stupidly EV
			double EV_inv = EV[i_kplus+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq];
			double candidate = llambda*profit[i_k+i_K*nk+i_s*nk*nK] + mmu*( llambda*(-effective_price)*(k_grid[i_kplus]-(1-p.ddelta)*k) - llambda*convexadj + p.bbeta*EV_inv ) + (1-mmu)*p.bbeta*EV_noinvest;
			if (candidate > rhsmax) {
				rhsmax         = candidate;
				koptind_active = i_kplus;
			};
		};

		// Find W and V finally
		Vplus[index] = rhsmax;
			koptindplus[index] = koptind_active;
		if (k_grid[koptind_active] != (1-p.ddelta)*k) {
			active[index]      = 1;
			kopt[index]        = k_grid[koptind_active];
		} else {
			active[index]      = 0;
			kopt[index]        = (1-p.ddelta)*k;
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
	p.eeta         = 0.0;
	p.Pssigmax[0] = 0.95; p.Pssigmax[2] = 0.05;
	p.Pssigmax[1] = 0.08; p.Pssigmax[3] = 0.92;

	// Create all STATE, SHOCK grids here
	h_vec_d h_k_grid(nk,0.0);
	h_vec_d h_K_grid(nK,0.0);
	h_vec_d h_z_grid(nz,0.0);
	h_vec_d h_x_grid(nx,0.0);
	h_vec_d h_ssigmax_grid(nssigmax,0.0);
	h_vec_d h_q_grid(nq,0.0);
	h_vec_d h_markup_grid(nmarkup,0.0);
	h_vec_d h_logZ(nz,0.0);
	h_vec_d h_logX(nx,0.0);
	h_vec_d h_PZ(nz*nz, 0.0);
	h_vec_d h_PX_low(nx*nx, 0.0);
	h_vec_d h_PX_high(nx*nx, 0.0);
	h_vec_d h_P(ns*ns, 0.0);
	h_vec_d h_V(nk*ns*nK*nq,0.0);
	h_vec_d h_Vplus(nk*ns*nK*nq,0.0);
	h_vec_d h_W(nk*ns*nK*nq,0.0);
	h_vec_d h_U(nk*ns*nK,0.0);
	h_vec_d h_EV(nk*ns*nK*nq,0.0);
	h_vec_d h_profit(nk*ns*nK,0.0);
	h_vec_i h_koptind(nk*ns*nK*nq,0);
	h_vec_i h_koptindplus(nk*ns*nK*nq,0);
	h_vec_d h_kopt(nk*ns*nK*nq,0.0);
	h_vec_i h_active(nk*ns*nK*nq,0);

	// load_vec(h_V,"./results/Vgrid.csv"); // in #include "cuda_helpers.h"

	// Create capital grid
	double maxK = 20.0;
	double minK = maxK*pow((1-p.ddelta),nk-1);
	for (int i_k = 0; i_k < nk; i_k++) {
		h_k_grid[i_k] = maxK*pow(1-p.ddelta,nk-1-i_k);
	};
	/* linspace(minK,maxK,nk,thrust::raw_pointer_cast(h_k_grid.data())); // in #include "cuda_helpers.h" */
	linspace(h_k_grid[0],h_k_grid[nk-1],nK,thrust::raw_pointer_cast(h_K_grid.data())); // in #include "cuda_helpers.h"

	// Create shocks grids and transition matrix
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
		int i_ssigmax = i_s/(nx*nz);
		int i_z       = (i_s-i_ssigmax*nx*nz)/(nx);
		int i_x       = (i_s-i_ssigmax*nx*nz-i_z*nx)/(1);
		for (int i_splus = 0; i_splus < ns; i_splus++) {
			int i_ssigmaxplus = i_splus/(nx*nz);
			int i_zplus       = (i_splus-i_ssigmaxplus*nx*nz)/(nx);
			int i_xplus       = (i_splus-i_ssigmaxplus*nx*nz-i_zplus*nx)/(1);
			if (i_ssigmaxplus==0) {
				h_P[i_s+i_splus*ns] = h_PX_low[i_x+i_xplus*nx]*h_PZ[i_z+i_zplus*nz]* p.Pssigmax[i_ssigmax+i_ssigmaxplus*nssigmax];
			} else {
				h_P[i_s+i_splus*ns] = h_PX_high[i_x+i_xplus*nx]*h_PZ[i_z+i_zplus*nz]*p.Pssigmax[i_ssigmax+i_ssigmaxplus*nssigmax];
			}
		};
	};

	// find cdf on host then transfer to device
	cudavec<double> CDF_z(nz*nz,0);                   pdf2cdf(h_PZ_ptr,nz,CDF_z.hptr);                   CDF_z.h2d();
	cudavec<double> CDF_ssigmax(nssigmax*nssigmax,0); pdf2cdf(p.Pssigmax,nssigmax,CDF_ssigmax.hptr); CDF_ssigmax.h2d();
	cudavec<double> CDF_x_low(nx*nx,0);               pdf2cdf(h_PX_low_ptr,nx,CDF_x_low.hptr);           CDF_x_low.h2d();
	cudavec<double> CDF_x_high(nx*nx,0);              pdf2cdf(h_PX_high_ptr,nx,CDF_x_high.hptr);         CDF_x_high.h2d();

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
	d_vec_i d_koptindplus  = h_koptindplus;
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
	double* d_EV_ptr           = raw_pointer_cast(d_EV.data());
	double* d_W_ptr            = raw_pointer_cast(d_W.data());
	double* d_Vplus_ptr        = raw_pointer_cast(d_Vplus.data());
	double* d_U_ptr            = raw_pointer_cast(d_U.data());
	double* d_P_ptr            = raw_pointer_cast(d_P.data());
	double* d_kopt_ptr         = raw_pointer_cast(d_kopt.data());
	int* d_koptind_ptr         = raw_pointer_cast(d_koptind.data());
	int* d_koptindplus_ptr     = raw_pointer_cast(d_koptindplus.data());
	int* d_active_ptr          = raw_pointer_cast(d_active.data());

	// Firstly a virtual index array from 0 to nk*nk*nz
	thrust::counting_iterator<int> begin(0);
	thrust::counting_iterator<int> end(nk*ns*nK*nq);
	thrust::counting_iterator<int> begin_noq(0);
	thrust::counting_iterator<int> end_noq(nk*ns*nK);
	thrust::counting_iterator<int> begin_sim(0);
	thrust::counting_iterator<int> end_sim(SIMULPERIOD);

	// generate aggregate shocks
	cudavec<double> innov_z(SIMULPERIOD);
	cudavec<double> innov_ssigmax(SIMULPERIOD);
	cudavec<double> innov_x(nhousehold*SIMULPERIOD);
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	curandGenerateUniformDouble(gen, innov_z.dptr,       SIMULPERIOD);
	curandGenerateUniformDouble(gen, innov_ssigmax.dptr, SIMULPERIOD);
	curandGenerateUniformDouble(gen, innov_x.dptr,       nhousehold*SIMULPERIOD);
	innov_z.d2h();
	innov_ssigmax.d2h();
	curandDestroyGenerator(gen);

	// simulate z and ssigmax index beforehand
	cudavec<double> z_sim(SIMULPERIOD);
	cudavec<double> ssigmax_sim(SIMULPERIOD);
	z_sim.hptr[0] = (nz-1)/2;
	ssigmax_sim.hptr[0] = (nssigmax-1)/2;
	for (int t = 1; t < SIMULPERIOD; t++) {
		z_sim.hptr[t]       = markovdiscrete(z_sim.hptr[t-1],CDF_z.hptr,nz,innov_z.hptr[t]);
		ssigmax_sim.hptr[t] = markovdiscrete(ssigmax_sim.hptr[t-1],CDF_ssigmax.hptr,nssigmax,innov_ssigmax.hptr[t]);
	};
	z_sim.h2d();
	ssigmax_sim.h2d();

    // Create Timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    // Start Timer
	cudaEventRecord(start,NULL);

	// Prepare for cuBLAS things
	cublasHandle_t handle;
	cublasCreate(&handle);
	const double alpha = 1.0;
	const double beta = 0.0;

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

	// vfi begins
	double diff = 10;  int iter = 0; int consec = 0;
	while ((diff>tol)&&(iter<maxiter)&&(consec<20)){
		// Find EV = V*tran(P), EV is EV(i_kplus,i_Kplus,i_qplus,i_s)
		cublasDgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			nk*nK*nq,
			ns,
			ns,
			&alpha,
			d_V_ptr,
			nk*nK*nq,
			d_P_ptr,
			ns,
			&beta,
			d_EV_ptr,
			nk*nK*nq
		);

		// find W/V currently
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
				d_EV_ptr,
				d_W_ptr,
				d_U_ptr,
				d_V_ptr,
				d_Vplus_ptr,
				d_kopt_ptr,
				d_active_ptr,
				d_koptindplus_ptr,
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

		// Check how many consecutive periods policy hasn't change
		int policy_diff = thrust::transform_reduce(
			thrust::make_zip_iterator(thrust::make_tuple(d_koptind.begin(),d_koptindplus.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_koptind.end()  ,d_koptindplus.end())),
			myDist(),
			0.0,
			thrust::maximum<int>()
		);
		if (policy_diff == 0) {
			consec++;
		} else {
			consec = 0;
		};


		std::cout << "diff is: "<< diff << std::endl;
		std::cout << "consec is: "<< consec << std::endl;

		// update correspondence
		d_V       = d_Vplus;
		d_koptind = d_koptindplus;

		std::cout << ++iter << std::endl;
		std::cout << "=====================" << std::endl;
	};
	// VFI ends

	// simulation



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
	h_kopt = d_kopt;
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

	// to be safe destroy cuBLAS handle
	cublasDestroy(handle);

	return 0;
}
