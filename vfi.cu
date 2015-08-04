#define nk 200
#define nx 9
#define nz 9
#define nssigmax 2
#define ns nx*nz*nssigmax
#define nK 50
#define nq 50
#define nmarkup 50
#define tauchenwidth 3.0
#define tol 1e-4
#define outertol 1e-4
#define damp 0.5
#define maxconsec 20
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
#include <curand.h>

// Includes, my own creation
#include "cudatools/include/cudatools.hpp"

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
			double convexadj = p.eeta*(k_grid[i_kplus]-(1-p.ddelta)*k)*(k_grid[i_kplus]-(1-p.ddelta)*k)/k;
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
		if (k_grid[koptind_active] > (1-p.ddelta)*k) {
			active[index]      = 1;
			kopt[index]        = k_grid[koptind_active];
		} else {
			active[index]      = 0;
			kopt[index]        = (1-p.ddelta)*k;
		};
	};
};

// finds profit generated from each household i at time t
struct profitfromhh {
	// data members
	double* kopt;
	int*    active;
	double* k_grid;
	double  q;
	double  w;
	double  mmu;
	double* matchshock;
	int     Kind;
	int     qind;
	int     zind;
	int     ssigmaxind;
	int*    kind_sim;
	int*    xind_sim;
	para    p;
	double* profit_temp;

	// constructor
	__host__ __device__
	profitfromhh(
		double* kopt_ptr,
		int*    active_ptr,
		double* k_grid_ptr,
		double  _q,
		double  _w,
		double  _mmu,
		double* matchshock_ptr,
		int     _Kind,
		int     _qind,
		int     _zind,
		int     _ssigmaxind,
		int*    kind_sim_ptr,
		int*    xind_sim_ptr,
		para    _p,
		double* profit_temp_ptr
	) {
		kopt        = kopt_ptr;
		active      = active_ptr;
		k_grid      = k_grid_ptr;
		q           = _q;
		w           = _w;
		mmu         = _mmu;
		matchshock  = matchshock_ptr;
		Kind        = _Kind;
		qind        = _qind;
		zind        = _zind;
		ssigmaxind  = _ssigmaxind;
		kind_sim       = kind_sim_ptr;
		xind_sim       = xind_sim_ptr;
		profit_temp = profit_temp_ptr;
		p           = _p;
	}

	// operator to find profit from each household
	__host__ __device__
	void operator() (int index) {
		int kind = kind_sim[index];
		int xind = xind_sim[index];
		int i_s  = xind + zind*nx + ssigmaxind*nx*nz;
		int i_state = kind + Kind*nk + qind*nk*nK + i_s*nk*nK*nq;
		if (matchshock[index] < mmu) {
			profit_temp[index] = (q-p.MC)*active[i_state]*(kopt[i_state]-(1-p.ddelta)*k_grid[kind])/nhousehold;
		} else {
			profit_temp[index] = 0;
		};
	};
};

// finds profit generated from each household i at time t
struct outputfromhh {
	// data members
	double* k_grid;
	double* x_grid;
	double* z_grid;
	double  w;
	int     zind;
	int*    kind_sim;
	int*    xind_sim;
	para    p;
	double* outputlist;

	// constructor
	__host__ __device__
	outputfromhh(
		double* k_grid_ptr,
		double* x_grid_ptr,
		double* z_grid_ptr,
		double  _w,
		int     _zind,
		int*    kind_sim_ptr,
		int*    xind_sim_ptr,
		para    _p,
		double* outputlist_ptr
	) {
		k_grid     = k_grid_ptr;
		x_grid     = x_grid_ptr;
		z_grid     = z_grid_ptr;
		w          = _w;
		zind       = _zind;
		kind_sim   = kind_sim_ptr;
		xind_sim   = xind_sim_ptr;
		p          = _p;
		outputlist = outputlist_ptr;
	}

	// operator to find profit from each household
	__host__ __device__
	void operator() (int index) {
		int kind = kind_sim[index];
		double k = k_grid[kind];
		int xind = xind_sim[index];
		double x = x_grid[xind];
		double z = z_grid[zind];
		double l = pow( w/z/x/p.v/pow(k,p.aalpha), 1.0/(p.v-1) );
		outputlist[index] = z*x*pow(k,p.aalpha)*pow(l,p.v)/nhousehold;
	};
};

// finds profit generated from each household i at time t
struct simulateforward {
	// data members
	double* kopt;
	int*    koptind;
	int*    active;
	double* k_grid;
	double* z_grid;
	double* x_grid;
	double  q;
	double  w;
	double  mmu;
	double* matchshock;
	int     Kind;
	int     qind;
	int     zind;
	int     ssigmaxind;
	int*    kind_sim;
	double* k_sim;
	int*    xind_sim;
	int*    activelist;
	para    p;

	// constructor
	__host__ __device__
	simulateforward(
		double* kopt_ptr,
		int*    koptind_ptr,
		int*    active_ptr,
		double* k_grid_ptr,
		double* z_grid_ptr,
		double* x_grid_ptr,
		double  _q,
		double  _w,
		double  _mmu,
		double* matchshock_ptr,
		int     _Kind,
		int     _qind,
		int     _zind,
		int     _ssigmaxind,
		int*    kind_sim_ptr,
		double* k_sim_ptr,
		int*    xind_sim_ptr,
		int*    activelist_ptr,
		para    _p
	) {
		kopt       = kopt_ptr;
		koptind    = koptind_ptr;
		active     = active_ptr;
		k_grid     = k_grid_ptr;
		z_grid     = z_grid_ptr;
		x_grid     = x_grid_ptr;
		q          = _q;
		w          = _w;
		mmu        = _mmu;
		matchshock = matchshock_ptr;
		Kind       = _Kind;
		qind       = _qind;
		zind       = _zind;
		ssigmaxind = _ssigmaxind;
		kind_sim   = kind_sim_ptr;
		k_sim      = k_sim_ptr;
		xind_sim   = xind_sim_ptr;
		activelist = activelist_ptr;
		p          = _p;
	}

	// operator to find profit from each household
	__host__ __device__
	void operator() (int index) {
		int kind    = kind_sim[index];
		double k    = k_grid[kind];
		int xind    = xind_sim[index];
		int i_s     = xind + zind*nx + ssigmaxind*nx*nz;
		int i_state = kind + Kind*nk + qind*nk*nK + i_s*nk*nK*nq;
		if (matchshock[index] < mmu) {
			kind_sim[index+nhousehold] = active[i_state]*koptind[i_state];
			k_sim[index+nhousehold] = k_grid[kind_sim[index+nhousehold]];
		} else {
			int noinvest_ind = fit2grid((1-p.ddelta)*k,nk,k_grid);
			kind_sim[index+nhousehold] = noinvest_ind;
			k_sim[index+nhousehold] = k_grid[kind_sim[index+nhousehold]];
		};
		activelist[index] = active[i_state];
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

	load_vec(h_V,"./results/Vgrid.csv"); // in #include "cuda_helpers.h"

	// Create capital grid
	double maxK = 100.0;
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
	double minq = 0.2;
	double maxq = 15.0;
	double minmarkup = 1.0;
	double maxmarkup = 1.3;
	linspace(minq,maxq,nq,thrust::raw_pointer_cast(h_q_grid.data())); // in #include "cuda_helpers.h"
	linspace(minmarkup,maxmarkup,nmarkup,thrust::raw_pointer_cast(h_markup_grid.data())); // in #include "cuda_helpers.h"

	// set initial agg rules
	aggrules r;
	r.pphi_qC = log((maxq)); // constant term
	r.pphi_qK = 0; // w.r.t agg K
	r.pphi_qz = 0; // w.r.t agg TFP
	r.pphi_qssigmax = 0; // w.r.t uncertainty
	r.pphi_KK = 0.99;
	r.pphi_KC = log((maxK+minK)/2.0);
	r.pphi_Kz = 0;
	r.pphi_Kssigmax = 0;// Aggregate Law of motion for aggregate capital
	r.pphi_CC = log(2);
	r.pphi_CK = 0.0;
	r.pphi_Cz = 0.0;
	r.pphi_Cssigmax = 0.0;
	r.pphi_tthetaC = log(0.95); // tightness ratio depends on q
	r.pphi_tthetaK = 0.0;
	r.pphi_tthetaz = 0.0;
	r.pphi_tthetassigmax = 0.0;
	r.pphi_tthetaq = 0.1;// lower q -- more firms invest -- lower ttheta
	r.loadfromfile("./results/aggrules.csv");

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
	thrust::counting_iterator<int> begin_hh(0);
	thrust::counting_iterator<int> end_hh(nhousehold);

	// generate aggregate shocks
	cudavec<double> innov_z(SIMULPERIOD);
	cudavec<double> innov_ssigmax(SIMULPERIOD);
	cudavec<double> innov_x(nhousehold*SIMULPERIOD);
	cudavec<double> innov_match(nhousehold*SIMULPERIOD);
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	curandGenerateUniformDouble(gen, innov_z.dptr,       SIMULPERIOD);
	curandGenerateUniformDouble(gen, innov_ssigmax.dptr, SIMULPERIOD);
	curandGenerateUniformDouble(gen, innov_x.dptr,       nhousehold*SIMULPERIOD);
	curandGenerateUniformDouble(gen, innov_match.dptr,   nhousehold*SIMULPERIOD);
	innov_z.d2h();
	innov_ssigmax.d2h();
	innov_x.d2h();
	curandDestroyGenerator(gen);

	// simulate z and ssigmax index beforehand
	cudavec<int>    zind_sim(SIMULPERIOD,(nz-1)/2);
	cudavec<int>    ssigmaxind_sim(SIMULPERIOD,(nssigmax-1)/2);
	cudavec<int>    xind_sim(nhousehold*SIMULPERIOD,(nx-1)/2);
	cudavec<double> z_sim(SIMULPERIOD,h_z_grid[(nz-1)/2]);
	cudavec<double> ssigmax_sim(SIMULPERIOD,h_ssigmax_grid[(nssigmax-1)/2]);
	for (int t = 1; t < SIMULPERIOD; t++) {
		zind_sim.hptr[t]       = markovdiscrete(zind_sim.hptr[t-1],CDF_z.hptr,nz,innov_z.hptr[t]);
		ssigmaxind_sim.hptr[t] = markovdiscrete(ssigmaxind_sim.hptr[t-1],CDF_ssigmax.hptr,nssigmax,innov_ssigmax.hptr[t]);
		z_sim.hptr[t] = h_z_grid[zind_sim.hptr[t]];
		ssigmax_sim.hptr[t] = h_ssigmax_grid[ssigmaxind_sim.hptr[t]];
		for (int i_household = 0; i_household < nhousehold; i_household++) {
			if (ssigmax_sim.hptr[t-1]==0) {
				xind_sim[i_household+t*nhousehold] = markovdiscrete(xind_sim[i_household+(t-1)*nhousehold],CDF_x_low.hptr,nx,innov_x.hptr[i_household+t*nhousehold]);
			};
			if (ssigmax_sim.hptr[t-1]==1) {
				xind_sim[i_household+t*nhousehold] = markovdiscrete(xind_sim[i_household+(t-1)*nhousehold],CDF_x_high.hptr,nx,innov_x.hptr[i_household+t*nhousehold]);
			};
		};
	};
	zind_sim.h2d();
	ssigmaxind_sim.h2d();
	xind_sim.h2d();
	display_vec(CDF_z);
	display_vec(CDF_ssigmax);

	// Prepare for cuBLAS things
	cublasHandle_t handle;
	cublasCreate(&handle);
	const double alpha = 1.0;
	const double beta = 0.0;

	// intialize simulaiton records
	cudavec<double> K_sim(SIMULPERIOD,(h_K_grid[0]+h_K_grid[nK-1])/2);
	cudavec<double> Kind_sim(SIMULPERIOD,(nK-1)/2);
	cudavec<double> k_sim(nhousehold*SIMULPERIOD,h_k_grid[(nk-1)/2]);
	cudavec<int>    kind_sim(nhousehold*SIMULPERIOD,(nk-1)/2);
	cudavec<double> profit_temp(nhousehold,0.0);
	cudavec<double> outputlist(nhousehold,0.0);
	cudavec<int>    activelist(nhousehold,0.0);
	cudavec<int>    qind_sim(SIMULPERIOD,(nq-1)/2);
	cudavec<double> q_sim(SIMULPERIOD,h_q_grid[(nq-1)/2]);
	cudavec<double> C_sim(SIMULPERIOD,0.0);
	cudavec<double> ttheta_sim(SIMULPERIOD,0.0);

	double outer_Rsq=0.0;
	while (outer_Rsq < 0.9) {

		// Create Timer
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		// Start Timer
		cudaEventRecord(start,NULL);

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
		while ((diff>tol)&&(iter<maxiter)&&(consec<maxconsec)){
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
		// VFI ends //

		// simulation given policies
		for (unsigned int t = 0; t < SIMULPERIOD; t++) {
			// find aggregate K from distribution of k
			K_sim[t] =  thrust::reduce(k_sim.dvec.begin()+t*nhousehold, k_sim.dvec.begin()+nhousehold+t*nhousehold, (double) 0, thrust::plus<double>())/double(nhousehold);
			Kind_sim[t] = fit2grid(K_sim[t],nK,thrust::raw_pointer_cast(h_K_grid.data()));

			// find current wage from aggregate things
			double C = exp( r.pphi_CC + r.pphi_CK*log(K_sim.hvec[t]) + r.pphi_Cssigmax*log(ssigmax_sim.hvec[t]) + r.pphi_Cz*log(z_sim.hvec[t]) );
			double w = p.ppsi_n*C;
			double* matchshock_ptr = thrust::raw_pointer_cast(innov_match.dvec.data()+t*nhousehold);
			int* kindlist_ptr      = thrust::raw_pointer_cast(kind_sim.dvec.data()+t*nhousehold);
			double* klist_ptr      = thrust::raw_pointer_cast(k_sim.dvec.data()+t*nhousehold);
			int* xindlist_ptr      = thrust::raw_pointer_cast(xind_sim.dvec.data()+t*nhousehold);

			// compute profit from each hh
			thrust::for_each(
				begin_hh,
				end_hh,
				outputfromhh(
					d_k_grid_ptr,
					d_x_grid_ptr,
					d_z_grid_ptr,
					w,
					zind_sim.hvec[t],
					kindlist_ptr,
					xindlist_ptr,
					p,
					outputlist.dptr
				)
			);
			double output = thrust::reduce(outputlist.dvec.begin(), outputlist.dvec.end(), (double) 0, thrust::plus<double>());

			// given markup find optimal price for monopolist
			double profitmax = -9999999;
			int i_qmax = 0;
			double inv = 0;
			for (unsigned int i_q = 0; i_q < nq; i_q++) {
				// find current variables
				double q           = h_q_grid[i_q];
				double ttheta_temp = exp( r.pphi_tthetaC + r.pphi_tthetaK*log(K_sim.hvec[t]) + r.pphi_tthetassigmax*log(ssigmax_sim.hvec[t]) + r.pphi_tthetaz*log(z_sim.hvec[t]) + r.pphi_tthetaq*log(q) );
				double mmu    = p.aalpha0*pow(ttheta_temp,p.aalpha1);

				// compute profit from each hh
				thrust::for_each(
					begin_hh,
					end_hh,
					profitfromhh(
						d_kopt_ptr,
						d_active_ptr,
						d_k_grid_ptr,
						q,
						w,
						mmu,
						matchshock_ptr,
						Kind_sim[t],
						i_q,
						zind_sim.hvec[t],
						ssigmaxind_sim.hvec[t],
						kindlist_ptr,
						xindlist_ptr,
						p,
						profit_temp.dptr
					)
				);

				// sum over profit to find total profit
				double totprofit = thrust::reduce(profit_temp.dvec.begin(), profit_temp.dvec.end(), (double) 0, thrust::plus<double>());
				double inv = totprofit/(q-p.MC);  ///< note that profit is already adjusted with 1/nhousehold
				if (totprofit > profitmax && inv < output) {
					profitmax = totprofit;
					i_qmax    = i_q;
				};
			}
			qind_sim[t] = i_qmax;
			double qmax = h_q_grid[i_qmax];
			q_sim[t] = qmax;

			// evolution under qmax!
			double ttheta_temp = exp( r.pphi_tthetaC + r.pphi_tthetaK*log(K_sim.hvec[t]) + r.pphi_tthetassigmax*log(ssigmax_sim.hvec[t]) + r.pphi_tthetaz*log(z_sim.hvec[t]) + r.pphi_tthetaq*log(qmax) );
			double mmu_temp    = p.aalpha0*pow(ttheta_temp,p.aalpha1);
			thrust::for_each(
				begin_hh,
				end_hh,
				simulateforward(
					d_kopt_ptr,
					d_koptind_ptr,
					d_active_ptr,
					d_k_grid_ptr,
					d_z_grid_ptr,
					d_x_grid_ptr,
					qmax,
					w,
					mmu_temp,
					matchshock_ptr,
					Kind_sim[t],
					i_qmax,
					zind_sim.hvec[t],
					ssigmaxind_sim.hvec[t],
					kindlist_ptr,
					klist_ptr,
					xindlist_ptr,
					activelist.dptr,
					p
				)
			);

			// find aggregate C and active ttheta
			double C_temp = thrust::reduce(outputlist.dvec.begin(), outputlist.dvec.end(), (double) 0, thrust::plus<double>())/double(nhousehold) - inv;
			C_sim.hptr[t]        = (C_temp > 0.0) ? C_temp : 1e-15;
			int activecount = thrust::reduce(activelist.dvec.begin(), activelist.dvec.end(), (int) 0, thrust::plus<int>());
			if (activecount != 0) {
				ttheta_sim.hptr[t]   = double(nhousehold)/double(activecount);
			} else {
				ttheta_sim.hptr[t]   = 1234567789.0;
			};
		};

		// regression
		double bbeta[5];
		double* X[4];
		X[0] = K_sim.hptr;
		X[1] = z_sim.hptr;
		X[2] = ssigmax_sim.hptr;
		X[3] = q_sim.hptr;

		// save simulations
		save_vec(K_sim,"./results/K_sim.csv");             // in #include "cuda_helpers.h"
		save_vec(z_sim,"./results/z_sim.csv");             // in #include "cuda_helpers.h"
		save_vec(ssigmax_sim,"./results/ssigmax_sim.csv"); // in #include "cuda_helpers.h"
		save_vec(q_sim,"./results/q_sim.csv");       // in #include "cuda_helpers.h"
		save_vec(C_sim,"./results/C_sim.csv");       // in #include "cuda_helpers.h"
		save_vec(ttheta_sim,"./results/ttheta_sim.csv");       // in #include "cuda_helpers.h"

		// run each regression and report
		double Rsq_K = logOLS(K_sim.hptr+1,X,SIMULPERIOD-1,3,bbeta);
		r.pphi_KC = bbeta[0]; r.pphi_KK = bbeta[1]; r.pphi_Kz = bbeta[2]; r.pphi_Kssigmax = bbeta[3];
		printf("Rsq_K = %.4f, log(Kplus) = %.4f + %.4f * log(K) + %.4f * log(ssigmax)+%.4f * log(z).\n",Rsq_K,r.pphi_KC,r.pphi_KK,r.pphi_Kssigmax,r.pphi_Kz);

		double Rsq_q = logOLS(q_sim.hptr+1,X,SIMULPERIOD-1,3,bbeta);
		r.pphi_qC = bbeta[0]; r.pphi_qK = bbeta[1]; r.pphi_qz = bbeta[2]; r.pphi_qssigmax = bbeta[3];
		printf("Rsq_q = %.4f, log(qplus) = %.4f + %.4f * log(K) + %.4f * log(ssigmax)+%.4f * log(z).\n",Rsq_q,r.pphi_qC,r.pphi_qK,r.pphi_qssigmax,r.pphi_qz);

		double Rsq_C = logOLS(C_sim.hptr,X,SIMULPERIOD,3,bbeta);
		r.pphi_CC = bbeta[0]; r.pphi_CK = bbeta[1]; r.pphi_Cz = bbeta[2]; r.pphi_Cssigmax = bbeta[3];
		printf("Rsq_C = %.4f, log(C) = %.4f + %.4f * log(K) + %.4f * log(ssigmax)+%.4f * log(z).\n",Rsq_C,r.pphi_CC,r.pphi_CK,r.pphi_Cssigmax,r.pphi_Cz);

		double Rsq_ttheta = logOLS(ttheta_sim.hptr,X,SIMULPERIOD,4,bbeta);
		r.pphi_tthetaC = bbeta[0]; r.pphi_tthetaK = bbeta[1]; r.pphi_tthetaz = bbeta[2]; r.pphi_tthetassigmax = bbeta[3]; r.pphi_tthetaq = bbeta[4];
		printf("Rsq_ttheta = %.4f, log(ttheta) = %.4f + %.4f * log(K) + %.4f * log(ssigmax)+%.4f * log(z) + %.4f*log(q).\n",Rsq_ttheta,r.pphi_tthetaC,r.pphi_tthetaK,r.pphi_tthetassigmax,r.pphi_tthetaz,r.pphi_tthetaq);

		outer_Rsq =  min(min(Rsq_K,Rsq_q),min(Rsq_C,Rsq_ttheta));


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

		r.savetofile("./results/aggrules.csv");
		save_vec(h_K_grid,"./results/K_grid.csv");         // in #include "cuda_helpers.h"
		save_vec(h_k_grid,"./results/k_grid.csv");         // in #include "cuda_helpers.h"
		save_vec(h_V,"./results/Vgrid.csv");               // in #include "cuda_helpers.h"
		save_vec(h_active,"./results/active.csv");         // in #include "cuda_helpers.h"
		save_vec(h_koptind,"./results/koptind.csv");       // in #include "cuda_helpers.h"
		save_vec(h_kopt,"./results/kopt.csv");             // in #include "cuda_helpers.h"
		std::cout << "Policy functions output completed." << std::endl;
	}


	// Export parameters to MATLAB
	p.exportmatlab("./MATLAB/vfi_para.m");

	// to be safe destroy cuBLAS handle
	cublasDestroy(handle);

	return 0;
}
