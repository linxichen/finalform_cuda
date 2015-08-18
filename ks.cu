#define nk           250
#define nx           15
#define nz           2
#define nssigmax     2
#define ns           nx*nz*nssigmax
#define nK           30
#define nq           65
#define nmarkup      65
#define tauchenwidth 2.5
#define tol          1e-5
#define outertol     1e-5
#define damp         0.4
#define maxconsec    30
#define maxiter      2000
#define SIMULPERIOD  1000
#define burnin       100
#define nhousehold   10000
#define kwidth       1.5

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
		double C = r.getC(i_z,i_ssigmax,K);
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
		double C = r.getC(i_z,i_ssigmax,K);
		double Kplus = r.getKplus(i_z,i_ssigmax,K);
		double qplus = r.getqplus(i_z,i_ssigmax,K);
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
		double C     = r.getC(i_z,i_ssigmax,K);
		double Kplus = r.getKplus(i_z,i_ssigmax,K);
		double qplus = r.getqplus(i_z,i_ssigmax,K);
		double mmu   = r.getmmu(i_z,i_ssigmax,K,q);
		mmu = min(1.0,mmu);

		double llambda = 1/C;
		int i_Kplus = fit2grid(Kplus,nK,K_grid);
		int i_qplus = fit2grid(qplus,nq,q_grid);

		// find the indexes of (1-ddelta)*k
		int noinvest_ind = fit2grid((1-p.ddelta)*k,nk,k_grid);
		int i_left_noinv, i_right_noinv;
		// (1-ddelta)k>=maxK, then should use K[nk-2] as left point
		if (noinvest_ind == nk-1) {
			i_left_noinv = nk-2;
			i_right_noinv = nk-1;
		} else {
			i_left_noinv = noinvest_ind;
			i_right_noinv = noinvest_ind+1;
		};

		// find EV_noinvest
		double kplus_left_noinv  = k_grid[i_left_noinv];
		double kplus_right_noinv = k_grid[i_right_noinv];
		double EV_noinvest = linear_interp(
			(1-p.ddelta)*k,
			kplus_left_noinv,
			kplus_right_noinv,
			EV[i_left_noinv+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq],
			EV[i_right_noinv+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq]
		);

		// search through all positve investment level
		double rhsmax = -999999999999999;
		int koptind_active = 0;
		for (int i_kplus = 0; i_kplus < nk; i_kplus++) {
			double inv = k_grid[i_kplus]-(1-p.ddelta)*k;
			double convexadj = p.eeta*inv*inv/k;
			double efftv_q = (inv>=0) ? q : p.pphi*q;
			// compute kinda stupidly EV
			double EV_inv = EV[i_kplus+i_Kplus*nk+i_qplus*nk*nK+i_s*nk*nK*nq];
			double candidate
				= llambda*profit[i_k+i_K*nk+i_s*nk*nK]
				+ mmu*( llambda*(-efftv_q*inv - convexadj) + p.bbeta*EV_inv )
				+ (1-mmu)*p.bbeta*EV_noinvest;
			if (candidate > rhsmax) {
				rhsmax         = candidate;
				koptind_active = i_kplus;
			};
		};

		// I can find U here
		double U = llambda*profit[i_k+i_K*nk+i_s*nk*nK] + p.bbeta*EV_noinvest;

		// Find W and V finally
		if (rhsmax>U) {
			Vplus[index]       = rhsmax;
			koptindplus[index] = koptind_active;
			active[index]      = 1;
			kopt[index]        = k_grid[koptind_active];
		} else {
			Vplus[index]       = U;
			koptindplus[index] = noinvest_ind;
			active[index]      = 0;
			kopt[index]        = (1-p.ddelta)*k;
		}
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
		int t = index/nhousehold;
		int i_house = index - t*nhousehold;
		int kind = kind_sim[index];
		int xind = xind_sim[index];
		int i_s  = xind + zind*nx + ssigmaxind*nx*nz;
		int i_state = kind + Kind*nk + qind*nk*nK + i_s*nk*nK*nq;
		if (matchshock[index] < mmu) {
			double inv = kopt[i_state]-(1-p.ddelta)*k_grid[kind];
			profit_temp[i_house] = (q-w)*double(active[i_state])*inv/nhousehold;
		} else {
			profit_temp[i_house] = 0;
		};
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
	double* clist;
	int*    activelist;
	para    p;
	int     T;

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
		double* clist_ptr,
		int*    activelist_ptr,
		para    _p,
		int     _T
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
		clist      = clist_ptr;
		activelist = activelist_ptr;
		p          = _p;
		T          = _T;
	}

	// operator to find profit from each household
	__host__ __device__
	void operator() (int index) {
		int t = index/nhousehold;
		int i_house = index - t*nhousehold;
		int kind    = kind_sim[index];
		double k    = k_grid[kind];
		int xind    = xind_sim[index];
		int i_s     = xind + zind*nx + ssigmaxind*nx*nz;
		int i_state = kind + Kind*nk + qind*nk*nK + i_s*nk*nK*nq;

		// find tomorrow's value if t < T
		if (t<T-1) {
			if (matchshock[index] < mmu && active[i_state] == 1) {
				kind_sim[i_house+(t+1)*nhousehold] = koptind[i_state];
				k_sim[i_house+(t+1)*nhousehold] = k_grid[koptind[i_state]];
			} else {
				int noinvest_ind = fit2grid((1-p.ddelta)*k,nk,k_grid);
				kind_sim[i_house+(t+1)*nhousehold] = noinvest_ind;
				k_sim[i_house+(t+1)*nhousehold] = (1-p.ddelta)*k;
			};
		};

		// find current variables
		double z = z_grid[zind];
		double x = x_grid[xind];
		double l = pow( w/z/x/p.v/pow(k,p.aalpha), 1.0/(p.v-1) );
		clist[i_house] = z*x*pow(k,p.aalpha)*pow(l,p.v);
		activelist[i_house] = active[i_state];
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

	// set device heap memeory size.
	/* cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1e8*sizeof(double)); */

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
	p.ppsi         = -1000000000000000000.00;
	p.rrhoz        = p.rrhox;
	p.ssigmaz      = 0.01;
	p.ssigmax_low  = 0.04;
	p.ssigmax_high = 0.04*3;
	p.ppsi_n       = 1;
	p.aalpha0      = 0.95;
	p.aalpha1      = 0.0;
	p.eeta         = 0.0;
	p.Pssigmax[0] = 0.95; p.Pssigmax[2] = 0.05;
	p.Pssigmax[1] = 0.08; p.Pssigmax[3] = 0.92;

	// Export parameters to MATLAB
	p.exportmatlab("./MATLAB/vfi_para.m");

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

	// obtain host pointers
	double* h_k_grid_ptr = thrust::raw_pointer_cast(h_k_grid.data());
	double* h_K_grid_ptr = thrust::raw_pointer_cast(h_K_grid.data());
	double* h_q_grid_ptr = thrust::raw_pointer_cast(h_q_grid.data());
	double* h_markup_grid_ptr = thrust::raw_pointer_cast(h_markup_grid.data());

	load_vec(h_V,"./results/Vgrid.csv"); // in #include "cuda_helpers.h"

	// Create capital grid
	double maxK = 12.0;
	double minK = 0.1;
	/* for (int i_k = 0; i_k < nk; i_k++) { */
	/* 	h_k_grid[i_k] = maxK*pow(1-p.ddelta,nk-1-i_k); */
	/* }; */
	linspace(minK,maxK,nk,h_k_grid_ptr);
	linspace(h_k_grid[0],h_k_grid[nk-1],nK,h_K_grid_ptr);

	// Create TFP grids and transition matrix
	h_ssigmax_grid[0] = p.ssigmax_low;
	h_ssigmax_grid[1] = p.ssigmax_high;
	double* h_logZ_ptr = thrust::raw_pointer_cast(h_logZ.data());
	double* h_PZ_ptr   = thrust::raw_pointer_cast(  h_PZ.data());
	/* tauchen(p.rrhoz, p.ssigmaz, h_logZ, h_PZ, tauchenwidth); */
	/* for (int i_z = 0; i_z < nz; i_z++) { */
	/* 	h_z_grid[i_z] = exp(h_logZ[i_z]); */
	/* }; */
	h_z_grid[0] = 0.99; h_z_grid[1] = 1.01;
	h_PZ[0] = 0.875;   h_PZ[2] = 1-0.875;
	h_PZ[1] = 1-0.875; h_PZ[3] = 0.875;

	// create idio prod grid and transition
	double* h_logX_ptr    = thrust::raw_pointer_cast(h_logX   .data());
	double* h_PX_low_ptr  = thrust::raw_pointer_cast(h_PX_low .data());
	double* h_PX_high_ptr = thrust::raw_pointer_cast(h_PX_high.data());
	tauchen(p.rrhox, p.ssigmax_high, h_logX, h_PX_high, tauchenwidth);
	tauchen_givengrid(p.rrhox, p.ssigmax_low,  h_logX, h_PX_low, tauchenwidth);
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
				h_P[i_s+i_splus*ns]
					= h_PX_low[i_x+i_xplus*nx]
					* h_PZ[i_z+i_zplus*nz]
					* p.Pssigmax[i_ssigmax+i_ssigmaxplus*nssigmax];
			} else {
				h_P[i_s+i_splus*ns]
					= h_PX_high[i_x+i_xplus*nx]
					* h_PZ[i_z+i_zplus*nz]
					* p.Pssigmax[i_ssigmax+i_ssigmaxplus*nssigmax];
			}
		};
	};

	// find cdf on host then transfer to device
	cudavec<double> CDF_z(       nz*nz,             0);
	cudavec<double> CDF_ssigmax( nssigmax*nssigmax, 0);
	cudavec<double> CDF_x_low(   nx*nx,             0);
	cudavec<double> CDF_x_high(  nx*nx,             0);
	pdf2cdf(h_PZ_ptr,      nz,       CDF_z.hptr);
	pdf2cdf(p.Pssigmax,    nssigmax, CDF_ssigmax.hptr);
	pdf2cdf(h_PX_low_ptr,  nx,       CDF_x_low.hptr);
	pdf2cdf(h_PX_high_ptr, nx,       CDF_x_high.hptr);
	CDF_z      .h2d();
	CDF_ssigmax.h2d();
	CDF_x_low  .h2d();
	CDF_x_high .h2d();

	// Create pricing grids
	double minq = 0.3;
	double maxq = 1.8;
	double minmarkup = 1.0;
	double maxmarkup = 1.3;
	linspace(minq,maxq,nq,h_q_grid_ptr);
	linspace(minmarkup,maxmarkup,nmarkup,h_markup_grid_ptr);

	// At this point good idea to save all created grids
	save_vec(h_K_grid,"./results/K_grid.csv");
	save_vec(h_k_grid,"./results/k_grid.csv");
	save_vec(h_q_grid,"./results/q_grid.csv");
	save_vec(h_markup_grid,"./results/markup_grid.csv");
	save_vec(h_x_grid,"./results/x_grid.csv");
	save_vec(h_z_grid,"./results/z_grid.csv");
	save_vec(h_ssigmax_grid,"./results/ssigmax_grid.csv");
	save_vec(h_PZ,"./results/PZ.csv");
	save_vec(h_PX_low,"./results/PX_low.csv");
	save_vec(h_PX_high,"./results/PX_high.csv");
	save_vec(h_P,"./results/P.csv");

	// set initial agg rules
	aggrules r;
	r.pphi_qC = log(1.05); // constant term
	r.pphi_qzind = 0; // w.r.t agg TFP
	r.pphi_qssigmaxind = 0; // w.r.t uncertainty
	r.pphi_qssigmaxindzind = 0; // w.r.t uncertainty
	r.pphi_qK = 0; // w.r.t agg K
	r.pphi_qssigmaxindK = 0; // w.r.t agg K
	r.pphi_qzindK = 0; // w.r.t agg K
	r.pphi_qssigmaxindzindK = 0; // w.r.t agg K

	r.pphi_KC = log(6.5); // constant term
	r.pphi_Kzind = 0; // w.r.t agg TFP
	r.pphi_Kssigmaxind = 0; // w.r.t uncertainty
	r.pphi_Kssigmaxindzind = 0; // w.r.t uncertainty
	r.pphi_KK = 0.0; // w.r.t agg K
	r.pphi_KssigmaxindK = 0; // w.r.t agg K
	r.pphi_KzindK = 0; // w.r.t agg K
	r.pphi_KssigmaxindzindK = 0; // w.r.t agg K

	r.pphi_CC = log(1.1); //constant term
	r.pphi_Czind = 0; // w.r.t agg TFP
	r.pphi_Cssigmaxind = 0; // w.r.t uncertainty
	r.pphi_Cssigmaxindzind = 0; // w.r.t uncertainty
	r.pphi_CK = 0.0; // w.r.t agg K
	r.pphi_CssigmaxindK = 0; // w.r.t agg K
	r.pphi_CzindK = 0; // w.r.t agg K
	r.pphi_CssigmaxindzindK = 0; // w.r.t agg K

	r.pphi_mmuC = log(0.95); // constant term
	r.pphi_mmuzind = 0; // w.r.t agg TFP
	r.pphi_mmussigmaxind = 0; // w.r.t uncertainty
	r.pphi_mmussigmaxindzind = 0; // w.r.t uncertainty
	r.pphi_mmuK = 0.0; // w.r.t agg K
	r.pphi_mmussigmaxindK = 0; // w.r.t agg K
	r.pphi_mmuzindK = 0; // w.r.t agg K
	r.pphi_mmussigmaxindzindK = 0; // w.r.t agg K
	r.pphi_mmuq = 0.0;// lower q -- more firms invest -- lower mmu

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
	cudavec<int>    zind_sim(SIMULPERIOD,            (nz-1)/2);
	cudavec<int>    ssigmaxind_sim(SIMULPERIOD,      (nssigmax-1)/2);
	cudavec<int>    xind_sim(nhousehold*SIMULPERIOD, (nx-1)/2);
	cudavec<double> z_sim(SIMULPERIOD,               h_z_grid[(nz-1)/2]);
	cudavec<double> ssigmax_sim(SIMULPERIOD,         h_ssigmax_grid[(nssigmax-1)/2]);
	/* markovsimul(SIMULPERIOD,CDF_z.hptr,nz,innov_z.hptr,(nz-1)/2,zind_sim.hptr); */
	/* markovsimul(SIMULPERIOD,CDF_ssigmax.hptr,nssigmax,innov_ssigmax.hptr,(nssigmax-1)/2,ssigmaxind_sim.hptr); */
	for (int t = 1; t < SIMULPERIOD; t++) {
		zind_sim.hptr[t] = markovdiscrete(zind_sim.hptr[t-1],CDF_z.hptr,nz,innov_z.hptr[t]);
		ssigmaxind_sim.hptr[t] = markovdiscrete(ssigmaxind_sim.hptr[t-1],CDF_ssigmax.hptr,nssigmax,innov_ssigmax.hptr[t]);
		z_sim.hptr[t] = h_z_grid[zind_sim.hptr[t]];
		ssigmax_sim.hptr[t] = h_ssigmax_grid[ssigmaxind_sim.hptr[t]];
		for (int i_household = 0; i_household < nhousehold; i_household++) {
			if (ssigmaxind_sim.hptr[t-1]==0) {
				xind_sim[i_household+t*nhousehold] = markovdiscrete(xind_sim[i_household+(t-1)*nhousehold],CDF_x_low.hptr,nx,innov_x.hptr[i_household+t*nhousehold]);
			};
			if (ssigmaxind_sim.hptr[t-1]==1) {
				xind_sim[i_household+t*nhousehold] = markovdiscrete(xind_sim[i_household+(t-1)*nhousehold],CDF_x_high.hptr,nx,innov_x.hptr[i_household+t*nhousehold]);
			};
		};
	};
	zind_sim.h2d();
	ssigmaxind_sim.h2d();
	xind_sim.h2d();

	// Prepare for cuBLAS things
	cublasHandle_t handle;
	cublasCreate(&handle);
	const double alpha = 1.0;
	const double beta = 0.0;

	// intialize simulaiton records
	cudavec<double> K_sim(SIMULPERIOD,(h_K_grid[0]+h_K_grid[nK-1])/2);
	cudavec<int>    Kind_sim(SIMULPERIOD,(nK-1)/2);
	cudavec<double> k_sim(nhousehold*SIMULPERIOD,h_k_grid[(nk-1)/2]);
	cudavec<int>    kind_sim(nhousehold*SIMULPERIOD,(nk-1)/2);
	cudavec<double> profit_temp(nhousehold,0.0);
	cudavec<double> clist(nhousehold,0.0);
	cudavec<int>    activelist(nhousehold,0);
	cudavec<int>    qind_sim(SIMULPERIOD,(nq-1)/2);
	cudavec<double> q_sim(SIMULPERIOD,h_q_grid[(nq-1)/2]);
	cudavec<double> C_sim(SIMULPERIOD,0.0);
	cudavec<double> mmu_sim(SIMULPERIOD,0.0);
	k_sim.h2d();
	kind_sim.h2d();

	double outer_Rsq=0.0;
	while (outer_Rsq < 0.8) {

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
				0,
				thrust::maximum<int>()
			);
			if (policy_diff == 0) {
				consec++;
			} else {
				consec = 0;
			};

			if ( iter % 50 == 0) {
				std::cout << "diff is: "<< diff << std::endl;
				std::cout << "consec is: "<< consec << std::endl;
				std::cout << iter << std::endl;
				std::cout << "=====================" << std::endl;
			};

			// update correspondence
			d_V       = d_Vplus;
			d_koptind = d_koptindplus;
			iter++;
		};
		// VFI ends //

		// simulation given policies
		for (unsigned int t = 0; t < SIMULPERIOD; t++) {
			// find aggregate K from distribution of k
			double temp_K =
				thrust::reduce(
				k_sim.dvec.begin()+t*nhousehold,
				k_sim.dvec.begin()+t*nhousehold+nhousehold,
				(double) 0,
				thrust::plus<double>()
				);
			K_sim[t] = temp_K/double(nhousehold);
			Kind_sim[t] = fit2grid(K_sim[t],nK,h_K_grid_ptr);
			int ssigmaxind_lag  = (t==0) ? 0 : ssigmaxind_sim[t-1];

			// find current wage from aggregate things
			double C = r.getC(zind_sim[t],ssigmaxind_lag,K_sim[t]);
			double w = p.ppsi_n*C;

			// given markup find optimal price for monopolist
			double profitmax = -9999999;
			int i_qmax = 0;
			for (unsigned int i_markup = 0; i_markup < nmarkup; i_markup++) {
				// find current variables
				double q   = h_markup_grid[i_markup]*w;
				int i_q    = fit2grid(q, nq, h_q_grid_ptr);
				double mmu = r.getmmu(zind_sim[t], ssigmaxind_lag, K_sim[t], q);
				mmu = max(1.0,mmu);

				// compute profit from each hh
				thrust::for_each(
					begin_hh+t*nhousehold,
					end_hh+t*nhousehold,
					profitfromhh(
						d_kopt_ptr,
						d_active_ptr,
						d_k_grid_ptr,
						q,
						w,
						mmu,
						innov_match.dptr,
						Kind_sim[t],
						i_q,
						zind_sim.hvec[t],
						ssigmaxind_lag,
						kind_sim.dptr,
						xind_sim.dptr,
						p,
						profit_temp.dptr
					)
				);

				// sum over profit to find total profit
				double totprofit = thrust::reduce(
					profit_temp.dvec.begin(),
					profit_temp.dvec.end(),
					(double) 0,
					thrust::plus<double>()
				);
				if (totprofit > profitmax) {
					profitmax = totprofit;
					i_qmax    = i_q;
				};
			}
			qind_sim[t] = i_qmax;
			double qmax = h_q_grid[i_qmax];
			q_sim[t] = qmax;

			// evolution under qmax!
			double mmu = r.getmmu(zind_sim[t],ssigmaxind_lag,K_sim[t],qmax);
			thrust::for_each(
				begin_hh+t*nhousehold,
				end_hh+t*nhousehold,
				simulateforward(
					d_kopt_ptr,
					d_koptind_ptr,
					d_active_ptr,
					d_k_grid_ptr,
					d_z_grid_ptr,
					d_x_grid_ptr,
					qmax,
					w,
					mmu,
					innov_match.dptr,
					Kind_sim[t],
					i_qmax,
					zind_sim.hvec[t],
					ssigmaxind_lag,
					kind_sim.dptr,
					k_sim.dptr,
					xind_sim.dptr,
					clist.dptr,
					activelist.dptr,
					p,
					SIMULPERIOD
				)
			);

			// find aggregate C and active ttheta
			C_sim.hptr[t]        = thrust::reduce(clist.dvec.begin(), clist.dvec.end(), (double) 0, thrust::plus<double>())/double(nhousehold);
			int activecount = thrust::reduce(activelist.dvec.begin(), activelist.dvec.end(), (int) 0, thrust::plus<int>());
			double ttheta = double(activecount)/double(nhousehold);
			mmu_sim.hptr[t] = p.aalpha0*pow( max(ttheta,1.0/double(nhousehold)) , -p.aalpha1);
		};

		// save simulations
		save_vec(K_sim,"./results/K_sim.csv");             // in #include "cuda_helpers.h"
		save_vec(z_sim,"./results/z_sim.csv");             // in #include "cuda_helpers.h"
		save_vec(ssigmax_sim,"./results/ssigmax_sim.csv"); // in #include "cuda_helpers.h"
		save_vec(q_sim,"./results/q_sim.csv");       // in #include "cuda_helpers.h"
		save_vec(C_sim,"./results/C_sim.csv");       // in #include "cuda_helpers.h"
		save_vec(mmu_sim,"./results/mmu_sim.csv");       // in #include "cuda_helpers.h"

		// prepare regressors.
		cudavec<double> constant(SIMULPERIOD-burnin,1.0);
		cudavec<double> ssigmaxind(SIMULPERIOD-burnin,0);  /// remember we need to use lag ssigmax as uncertainty
		cudavec<double> zind(SIMULPERIOD-burnin,0);
		cudavec<double> ssigmaxindzind(SIMULPERIOD-burnin,0);  /// remember we need to use lag ssigmax as uncertainty
		cudavec<double> logK(SIMULPERIOD-burnin,0);
		cudavec<double> logq(SIMULPERIOD-burnin,1.0);
		cudavec<double> logC(SIMULPERIOD-burnin,1.0);
		cudavec<double> logmmu(SIMULPERIOD-burnin,1.0);
		cudavec<double> ssigmaxindK(SIMULPERIOD-burnin,0);
		cudavec<double> zindK(SIMULPERIOD-burnin,0);
		cudavec<double> ssigmaxindzindK(SIMULPERIOD-burnin,0);
		for (int t = 0; t < SIMULPERIOD-burnin; t++) {
			ssigmaxind[t] = ssigmaxind_sim[t-1+burnin];
			ssigmaxindzind[t] = ssigmaxind[t]*double(zind_sim[t+burnin]);
			logK[t] = log(K_sim[t+burnin]);
			logq[t] = log(q_sim[t+burnin]);
			logC[t] = log(C_sim[t+burnin]);
			logmmu[t] = log(mmu_sim[t+burnin]);
			ssigmaxindK[t] = ssigmaxind[t]*logK[t];
			zind[t] = double(zind_sim[t+burnin]);
			zindK[t] = zind[t]*logK[t];
			ssigmaxindzindK[t] = ssigmaxind[t]*zind[t]*logK[t];
		};
		double bbeta[9];
		double* X[9];
		X[0] = constant.hptr;
		X[1] = ssigmaxind.hptr;
		X[2] = zind.hptr;
		X[3] = ssigmaxindzind.hptr;
		X[4] = logK.hptr;
		X[5] = ssigmaxindK.hptr;
		X[6] = zindK.hptr;
		X[7] = ssigmaxindzindK.hptr;
		X[8] = logq.hptr;

		// run each regression and report
		double Rsq_K = levelOLS(logK.hptr+1,X,SIMULPERIOD-1-burnin,8,bbeta);
		r.pphi_KC               = (1.0-damp)*r.pphi_KC               + damp*bbeta[0];
		r.pphi_Kssigmaxind      = (1.0-damp)*r.pphi_Kssigmaxind      + damp*bbeta[1];
		r.pphi_Kzind            = (1.0-damp)*r.pphi_Kzind            + damp*bbeta[2];
		r.pphi_Kssigmaxindzind  = (1.0-damp)*r.pphi_Kssigmaxindzind  + damp*bbeta[3];
		r.pphi_KK               = (1.0-damp)*r.pphi_KK               + damp*bbeta[4];
		r.pphi_KssigmaxindK     = (1.0-damp)*r.pphi_KssigmaxindK     + damp*bbeta[5];
		r.pphi_KzindK           = (1.0-damp)*r.pphi_KzindK           + damp*bbeta[6];
		r.pphi_KssigmaxindzindK = (1.0-damp)*r.pphi_KssigmaxindzindK + damp*bbeta[7];
		printf("Rsq_K = %.4f, log(Kplus) = (%.2f+%.2f*ssigmaxind_lag+%.2f*zind+%.2f*ssigmaxind_lag*zind) + (%.2f+%.2f*ssigmaxind_lag+%.2f*zind+%.2f*ssigmaxind_lag*zind) * log(K) \n",Rsq_K,r.pphi_KC,r.pphi_Kssigmaxind,r.pphi_Kzind,r.pphi_Kssigmaxindzind,r.pphi_KK,r.pphi_KssigmaxindK,r.pphi_KzindK,r.pphi_KssigmaxindzindK);

		double Rsq_q = levelOLS(logq.hptr+1,X,SIMULPERIOD-1-burnin,8,bbeta);
		r.pphi_qC               = (1.0-damp)*r.pphi_qC               + damp*bbeta[0];
		r.pphi_qssigmaxind      = (1.0-damp)*r.pphi_qssigmaxind      + damp*bbeta[1];
		r.pphi_qzind            = (1.0-damp)*r.pphi_qzind            + damp*bbeta[2];
		r.pphi_qssigmaxindzind  = (1.0-damp)*r.pphi_qssigmaxindzind  + damp*bbeta[3];
		r.pphi_qK               = (1.0-damp)*r.pphi_qK               + damp*bbeta[4];
		r.pphi_qssigmaxindK     = (1.0-damp)*r.pphi_qssigmaxindK     + damp*bbeta[5];
		r.pphi_qzindK           = (1.0-damp)*r.pphi_qzindK           + damp*bbeta[6];
		r.pphi_qssigmaxindzindK = (1.0-damp)*r.pphi_qssigmaxindzindK + damp*bbeta[7];
		printf("Rsq_q = %.4f, log(qplus) = (%.2f+%.2f*ssigmaxind_lag+%.2f*zind+%.2f*ssigmaxind_lag*zind) + (%.2f+%.2f*ssigmaxind_lag+%.2f*zind+%.2f*ssigmaxind_lag*zind) * log(K) \n",Rsq_q,r.pphi_qC,r.pphi_qssigmaxind,r.pphi_qzind,r.pphi_qssigmaxindzind,r.pphi_qK,r.pphi_qssigmaxindK,r.pphi_qzindK,r.pphi_qssigmaxindzindK);

		double Rsq_C = levelOLS(logC.hptr,X,SIMULPERIOD-burnin,8,bbeta);
		r.pphi_CC               = (1.0-damp)*r.pphi_CC               + damp*bbeta[0];
		r.pphi_Cssigmaxind      = (1.0-damp)*r.pphi_Cssigmaxind      + damp*bbeta[1];
		r.pphi_Czind            = (1.0-damp)*r.pphi_Czind            + damp*bbeta[2];
		r.pphi_Cssigmaxindzind  = (1.0-damp)*r.pphi_Cssigmaxindzind  + damp*bbeta[3];
		r.pphi_CK               = (1.0-damp)*r.pphi_CK               + damp*bbeta[4];
		r.pphi_CssigmaxindK     = (1.0-damp)*r.pphi_CssigmaxindK     + damp*bbeta[5];
		r.pphi_CzindK           = (1.0-damp)*r.pphi_CzindK           + damp*bbeta[6];
		r.pphi_CssigmaxindzindK = (1.0-damp)*r.pphi_CssigmaxindzindK + damp*bbeta[7];
		printf("Rsq_C = %.4f, log(C) = (%.2f+%.2f*ssigmaxind_lag+%.2f*zind+%.2f*ssigmaxind_lag*zind) + (%.2f+%.2f*ssigmaxind_lag+%.2f*zind+%.2f*ssigmaxind_lag*zind) * log(K) \n",Rsq_C,r.pphi_CC,r.pphi_Cssigmaxind,r.pphi_Czind,r.pphi_Cssigmaxindzind,r.pphi_CK,r.pphi_CssigmaxindK,r.pphi_CzindK,r.pphi_CssigmaxindzindK);

		double Rsq_mmu = levelOLS(logmmu.hptr,X,SIMULPERIOD-burnin,9,bbeta);
		r.pphi_mmuC               = (1.0-damp)*r.pphi_mmuC               + damp*bbeta[0];
		r.pphi_mmussigmaxind      = (1.0-damp)*r.pphi_mmussigmaxind      + damp*bbeta[1];
		r.pphi_mmuzind            = (1.0-damp)*r.pphi_mmuzind            + damp*bbeta[2];
		r.pphi_mmussigmaxindzind  = (1.0-damp)*r.pphi_mmussigmaxindzind  + damp*bbeta[3];
		r.pphi_mmuK               = (1.0-damp)*r.pphi_mmuK               + damp*bbeta[4];
		r.pphi_mmussigmaxindK     = (1.0-damp)*r.pphi_mmussigmaxindK     + damp*bbeta[5];
		r.pphi_mmuzindK           = (1.0-damp)*r.pphi_mmuzindK           + damp*bbeta[6];
		r.pphi_mmussigmaxindzindK = (1.0-damp)*r.pphi_mmussigmaxindzindK + damp*bbeta[7];
		r.pphi_mmuq               = (1.0-damp)*r.pphi_mmuq               + damp*bbeta[8];
		printf("Rsq_mmu = %.4f, log(mmu) = (%.2f+%.2f*ssigmaxind_lag+%.2f*zind+%.2f*ssigmaxind_lag*zind) + (%.2f+%.2f*ssigmaxind_lag+%.2f*zind+%.2f*ssigmaxind_lag*zind) * log(K) + %.2f*log(q) \n",Rsq_mmu,r.pphi_mmuC,r.pphi_mmussigmaxind,r.pphi_mmuzind,r.pphi_mmussigmaxindzind,r.pphi_mmuK,r.pphi_mmussigmaxindK,r.pphi_mmuzindK,r.pphi_mmussigmaxindzindK,r.pphi_mmuq);

		outer_Rsq =  min(min(Rsq_K,Rsq_q),Rsq_C);

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
		/* save_vec(h_V,"./results/Vgrid.csv");               // in #include "cuda_helpers.h" */
		/* save_vec(h_active,"./results/active.csv");         // in #include "cuda_helpers.h" */
		/* save_vec(h_koptind,"./results/koptind.csv");       // in #include "cuda_helpers.h" */
		/* save_vec(h_kopt,"./results/kopt.csv");             // in #include "cuda_helpers.h" */
		std::cout << "Policy functions output completed." << std::endl;
	}

	///------------------------------------------------------------
	/// General IRF Creation
	///------------------------------------------------------------
	int irfperiods = 40;
	int nworlds    = 1000;


	// (REF path) initialze
	cudavec<int>    zind_sim_ref(nworlds*irfperiods,            1);
	cudavec<int>    ssigmaxind_sim_ref(nworlds*irfperiods,      0);
	cudavec<int>    xind_sim_ref(nhousehold*irfperiods, (nx-1)/2);
	cudavec<double> z_sim_ref(nworlds*irfperiods,               h_z_grid[0]);
	cudavec<double> ssigmax_sim_ref(nworlds*irfperiods,         h_ssigmax_grid[0]);
	cudavec<double> k_sim_ref(nhousehold*irfperiods,h_k_grid[(nk-1)/2]);
	cudavec<int>    kind_sim_ref(nhousehold*irfperiods,(nk-1)/2);
	cudavec<double> K_sim_ref(nworlds*irfperiods,K_sim[SIMULPERIOD-1]);
	cudavec<int>    Kind_sim_ref(nworlds*irfperiods,Kind_sim[SIMULPERIOD-1]);
	cudavec<int>    qind_sim_ref(nworlds*irfperiods,(nq-1)/2);
	cudavec<double> q_sim_ref(nworlds*irfperiods,h_q_grid[(nq-1)/2]);
	cudavec<double> C_sim_ref(nworlds*irfperiods,0.0);
	cudavec<double> ttheta_sim_ref(nworlds*irfperiods,0.0);
	cudavec<double> mmu_sim_ref(nworlds*irfperiods,0.0);

	// load initial state, z, ssigmax, and k,x distribution
	for (int i_world = 0; i_world < nworlds; i_world++) {
		int i_backward = SIMULPERIOD-1-i_world;
		zind_sim_ref[i_world+0*nworlds]       = zind_sim[i_backward];
		z_sim_ref[i_world+0*nworlds]          = z_sim[i_backward];
		ssigmaxind_sim_ref[i_world+0*nworlds] = ssigmaxind_sim[i_backward];
		ssigmax_sim_ref[i_world+0*nworlds]    = ssigmax_sim[i_backward];
		for (int i_hh = 0; i_hh < nhousehold; i_hh++) {
			xind_sim_ref[i_hh+0*nhousehold] = xind_sim[i_hh+i_backward*nhousehold];
			kind_sim_ref[i_hh+0*nhousehold] = kind_sim[i_hh+i_backward*nhousehold];
			k_sim_ref[i_hh+0*nhousehold]    = k_sim[i_hh+i_backward*nhousehold];
		};

		// generate innovation terms
		cudavec<double> eps_z(irfperiods);
		cudavec<double> eps_ssigmax(irfperiods);
		cudavec<double> eps_x(nhousehold*irfperiods);
		cudavec<double> eps_match(nhousehold*irfperiods);
		curandGenerator_t gen;
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen, int64_t(709394+i_world));
		curandGenerateUniformDouble(gen, eps_z.dptr,       irfperiods);
		curandGenerateUniformDouble(gen, eps_ssigmax.dptr, irfperiods);
		curandGenerateUniformDouble(gen, eps_x.dptr,       nhousehold*irfperiods);
		curandGenerateUniformDouble(gen, eps_match.dptr,   nhousehold*irfperiods);
		eps_z.d2h();
		eps_ssigmax.d2h();
		eps_x.d2h();
		curandDestroyGenerator(gen);

		// simulate z and ssigmax and x for each world
		for (int t = 1; t < irfperiods; t++) {
			zind_sim_ref.hptr[i_world+t*nworlds] =
				markovdiscrete(zind_sim_ref.hptr[i_world+(t-1)*nworlds],CDF_z.hptr,nz,eps_z.hptr[t]);
			ssigmaxind_sim_ref.hptr[i_world+t*nworlds] =
				markovdiscrete(ssigmaxind_sim_ref.hptr[i_world+(t-1)*nworlds],CDF_ssigmax.hptr,nssigmax,eps_ssigmax.hptr[t]);
			z_sim_ref.hptr[t] = h_z_grid[zind_sim_ref.hptr[t]];
			ssigmax_sim_ref.hptr[t] = h_ssigmax_grid[ssigmaxind_sim_ref.hptr[t]];
			for (int i_household = 0; i_household < nhousehold; i_household++) {
				if (ssigmaxind_sim_ref.hptr[t-1]==0) {
					xind_sim_ref[i_household+t*nhousehold] = markovdiscrete(xind_sim_ref[i_household+(t-1)*nhousehold],CDF_x_low.hptr,nx,eps_x.hptr[i_household+t*nhousehold]);
				};
				if (ssigmaxind_sim_ref.hptr[t-1]==1) {
					xind_sim_ref[i_household+t*nhousehold] =
						markovdiscrete(xind_sim_ref[i_household+(t-1)*nhousehold],CDF_x_high.hptr,nx,eps_x.hptr[i_household+t*nhousehold]);
				};
			};
		};
		zind_sim_ref.h2d();
		ssigmaxind_sim_ref.h2d();
		xind_sim_ref.h2d();
		k_sim_ref.h2d();
		kind_sim_ref.h2d();
		xind_sim_ref.h2d();
		/* printf("exo simul completed.\n"); */

		// intialize simulaiton records
		cudavec<double> profit_temp_ref(nhousehold,0.0);
		for (unsigned int t = 0; t < irfperiods; t++) {
			// find aggregate K from distribution of k
			double temp_K =
				thrust::reduce(
						k_sim_ref.dvec.begin()+t*nhousehold,
						k_sim_ref.dvec.begin()+t*nhousehold+nhousehold,
						(double) 0,
						thrust::plus<double>()
						);
			K_sim_ref[i_world+t*nworlds] = temp_K/double(nhousehold);
			Kind_sim_ref[i_world+t*nworlds] = fit2grid(K_sim_ref[i_world+t*nworlds],nK,h_K_grid_ptr);
			int ssigmaxind_lag  = (t==0) ? 0 :
				ssigmaxind_sim_ref[i_world+(t-1)*nworlds];

			// find current wage from aggregate things
			double C = r.getC(zind_sim_ref[i_world+t*nworlds],ssigmaxind_lag,K_sim_ref[i_world+t*nworlds]);
			double w = p.ppsi_n*C;

			// given markup find optimal price for monopolist
			double profitmax = -9999999;
			int i_qmax = 0;
			for (unsigned int i_markup = 0; i_markup < nmarkup; i_markup++) {
				// find current variables
				double q   = h_markup_grid[i_markup]*w;
				int i_q    = fit2grid(q, nq, h_q_grid_ptr);
				double mmu = r.getmmu(zind_sim_ref[i_world+t*nworlds], ssigmaxind_lag, K_sim_ref[i_world+t*nworlds], q);

				// compute profit from each hh
				thrust::for_each(
						begin_hh+t*nhousehold,
						end_hh+t*nhousehold,
						profitfromhh(
							d_kopt_ptr,
							d_active_ptr,
							d_k_grid_ptr,
							q,
							w,
							mmu,
							eps_match.dptr,
							Kind_sim_ref[i_world+t*nworlds],
							i_q,
							zind_sim_ref.hvec[i_world+t*nworlds],
							ssigmaxind_lag,
							kind_sim_ref.dptr,
							xind_sim_ref.dptr,
							p,
							profit_temp.dptr
							)
						);

				// sum over profit to find total profit
				double totprofit = thrust::reduce(profit_temp.dvec.begin(), profit_temp.dvec.end(), (double) 0, thrust::plus<double>());
				if (totprofit > profitmax) {
					profitmax = totprofit;
					i_qmax    = i_q;
				};
			}
			qind_sim_ref[i_world+t*nworlds] = i_qmax;
			double qmax = h_q_grid[i_qmax];
			q_sim_ref[i_world+t*nworlds] = qmax;
			/* printf("optimal q found\n"); */

			// evolution under qmax!
			double mmu = r.getmmu(zind_sim_ref[i_world+t*nworlds],ssigmaxind_lag,K_sim_ref[i_world+t*nworlds],qmax);
			thrust::for_each(
					begin_hh+t*nhousehold,
					end_hh+t*nhousehold,
					simulateforward(
						d_kopt_ptr,
						d_koptind_ptr,
						d_active_ptr,
						d_k_grid_ptr,
						d_z_grid_ptr,
						d_x_grid_ptr,
						qmax,
						w,
						mmu,
						eps_match.dptr,
						Kind_sim_ref[i_world+t*nworlds],
						i_qmax,
						zind_sim_ref.hvec[i_world+t*nworlds],
						ssigmaxind_lag,
						kind_sim_ref.dptr,
						k_sim_ref.dptr,
						xind_sim_ref.dptr,
						clist.dptr,
						activelist.dptr,
						p,
						irfperiods
						)
					);
			/* printf("simulate forward done\n"); */

			// find aggregate C and active ttheta
			C_sim_ref.hptr[i_world+t*nworlds] = thrust::reduce(clist.dvec.begin(), clist.dvec.end(), (double) 0, thrust::plus<double>())/double(nhousehold);
			int activecount = thrust::reduce(activelist.dvec.begin(), activelist.dvec.end(), (int) 0, thrust::plus<int>());
			double ttheta = double(activecount)/double(nhousehold);
			mmu_sim_ref.hptr[i_world+t*nworlds] = p.aalpha0*pow( max(ttheta,1.0/double(nhousehold)) , -p.aalpha1);
			ttheta_sim_ref.hptr[i_world+t*nworlds] = ttheta;
		/* printf("t=%d \n",t); */
		}
		/* printf("world=%d \n",i_world); */
	}

	// irf
	cudavec<int>    zind_sim_irf(nworlds*irfperiods,            1);
	cudavec<int>    ssigmaxind_sim_irf(nworlds*irfperiods,      0);
	cudavec<int>    xind_sim_irf(nhousehold*irfperiods, (nx-1)/2);
	cudavec<double> z_sim_irf(nworlds*irfperiods,               h_z_grid[0]);
	cudavec<double> ssigmax_sim_irf(nworlds*irfperiods,         h_ssigmax_grid[0]);
	cudavec<double> k_sim_irf(nhousehold*irfperiods,h_k_grid[(nk-1)/2]);
	cudavec<int>    kind_sim_irf(nhousehold*irfperiods,(nk-1)/2);
	cudavec<double> K_sim_irf(nworlds*irfperiods,K_sim[SIMULPERIOD-1]);
	cudavec<int>    Kind_sim_irf(nworlds*irfperiods,Kind_sim[SIMULPERIOD-1]);
	cudavec<int>    qind_sim_irf(nworlds*irfperiods,(nq-1)/2);
	cudavec<double> q_sim_irf(nworlds*irfperiods,h_q_grid[(nq-1)/2]);
	cudavec<double> C_sim_irf(nworlds*irfperiods,0.0);
	cudavec<double> ttheta_sim_irf(nworlds*irfperiods,0.0);
	cudavec<double> mmu_sim_irf(nworlds*irfperiods,0.0);

	// load initial state, z, ssigmax, and k,x distribution
	for (int i_world = 0; i_world < nworlds; i_world++) {
		int i_backward = SIMULPERIOD-1-i_world;
		zind_sim_irf[i_world+0*nworlds]       = zind_sim[i_backward];
		z_sim_irf[i_world+0*nworlds]          = z_sim[i_backward];
		ssigmaxind_sim_irf[i_world+0*nworlds] = 1;
		ssigmax_sim_irf[i_world+0*nworlds]    = h_ssigmax_grid[1];
		for (int i_hh = 0; i_hh < nhousehold; i_hh++) {
			xind_sim_irf[i_hh+0*nhousehold] = xind_sim[i_hh+i_backward*nhousehold];
			kind_sim_irf[i_hh+0*nhousehold] = kind_sim[i_hh+i_backward*nhousehold];
			k_sim_irf[i_hh+0*nhousehold]    = k_sim[i_hh+i_backward*nhousehold];
		};

		// generate innovation terms
		cudavec<double> eps_z(irfperiods);
		cudavec<double> eps_ssigmax(irfperiods);
		cudavec<double> eps_x(nhousehold*irfperiods);
		cudavec<double> eps_match(nhousehold*irfperiods);
		curandGenerator_t gen;
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen, int64_t(709394+i_world));
		curandGenerateUniformDouble(gen, eps_z.dptr,       irfperiods);
		curandGenerateUniformDouble(gen, eps_ssigmax.dptr, irfperiods);
		curandGenerateUniformDouble(gen, eps_x.dptr,       nhousehold*irfperiods);
		curandGenerateUniformDouble(gen, eps_match.dptr,   nhousehold*irfperiods);
		eps_z.d2h();
		eps_ssigmax.d2h();
		eps_x.d2h();
		curandDestroyGenerator(gen);

		// simulate z and ssigmax and x for each world
		for (int t = 1; t < irfperiods; t++) {
			zind_sim_irf.hptr[i_world+t*nworlds] =
				markovdiscrete(zind_sim_irf.hptr[i_world+(t-1)*nworlds],CDF_z.hptr,nz,eps_z.hptr[t]);
			ssigmaxind_sim_irf.hptr[i_world+t*nworlds] =
				markovdiscrete(ssigmaxind_sim_irf.hptr[i_world+(t-1)*nworlds],CDF_ssigmax.hptr,nssigmax,eps_ssigmax.hptr[t]);
			z_sim_irf.hptr[t] = h_z_grid[zind_sim_irf.hptr[t]];
			ssigmax_sim_irf.hptr[t] = h_ssigmax_grid[ssigmaxind_sim_irf.hptr[t]];
			for (int i_household = 0; i_household < nhousehold; i_household++) {
				if (ssigmaxind_sim_irf.hptr[t-1]==0) {
					xind_sim_irf[i_household+t*nhousehold] = markovdiscrete(xind_sim_irf[i_household+(t-1)*nhousehold],CDF_x_low.hptr,nx,eps_x.hptr[i_household+t*nhousehold]);
				};
				if (ssigmaxind_sim_irf.hptr[t-1]==1) {
					xind_sim_irf[i_household+t*nhousehold] =
						markovdiscrete(xind_sim_irf[i_household+(t-1)*nhousehold],CDF_x_high.hptr,nx,eps_x.hptr[i_household+t*nhousehold]);
				};
			};
		};
		zind_sim_irf.h2d();
		ssigmaxind_sim_irf.h2d();
		xind_sim_irf.h2d();
		k_sim_irf.h2d();
		kind_sim_irf.h2d();
		xind_sim_irf.h2d();
		/* printf("exo simul completed.\n"); */

		// intialize simulaiton records
		cudavec<double> profit_temp_irf(nhousehold,0.0);
		for (unsigned int t = 0; t < irfperiods; t++) {
			// find aggregate K from distribution of k
			double temp_K =
				thrust::reduce(
						k_sim_irf.dvec.begin()+t*nhousehold,
						k_sim_irf.dvec.begin()+t*nhousehold+nhousehold,
						(double) 0,
						thrust::plus<double>()
						);
			K_sim_irf[i_world+t*nworlds] = temp_K/double(nhousehold);
			Kind_sim_irf[i_world+t*nworlds] = fit2grid(K_sim_irf[i_world+t*nworlds],nK,h_K_grid_ptr);
			int ssigmaxind_lag  = (t==0) ? 0 :
				ssigmaxind_sim_irf[i_world+(t-1)*nworlds];

			// find current wage from aggregate things
			double C = r.getC(zind_sim_irf[i_world+t*nworlds],ssigmaxind_lag,K_sim_irf[i_world+t*nworlds]);
			double w = p.ppsi_n*C;

			// given markup find optimal price for monopolist
			double profitmax = -9999999;
			int i_qmax = 0;
			for (unsigned int i_markup = 0; i_markup < nmarkup; i_markup++) {
				// find current variables
				double q   = h_markup_grid[i_markup]*w;
				int i_q    = fit2grid(q, nq, h_q_grid_ptr);
				double mmu = r.getmmu(zind_sim_irf[i_world+t*nworlds], ssigmaxind_lag, K_sim_irf[i_world+t*nworlds], q);

				// compute profit from each hh
				thrust::for_each(
						begin_hh+t*nhousehold,
						end_hh+t*nhousehold,
						profitfromhh(
							d_kopt_ptr,
							d_active_ptr,
							d_k_grid_ptr,
							q,
							w,
							mmu,
							eps_match.dptr,
							Kind_sim_irf[i_world+t*nworlds],
							i_q,
							zind_sim_irf.hvec[i_world+t*nworlds],
							ssigmaxind_lag,
							kind_sim_irf.dptr,
							xind_sim_irf.dptr,
							p,
							profit_temp.dptr
							)
						);

				// sum over profit to find total profit
				double totprofit = thrust::reduce(profit_temp.dvec.begin(), profit_temp.dvec.end(), (double) 0, thrust::plus<double>());
				if (totprofit > profitmax) {
					profitmax = totprofit;
					i_qmax    = i_q;
				};
			}
			qind_sim_irf[i_world+t*nworlds] = i_qmax;
			double qmax = h_q_grid[i_qmax];
			q_sim_irf[i_world+t*nworlds] = qmax;
			/* printf("optimal q found\n"); */

			// evolution under qmax!
			double mmu = r.getmmu(zind_sim_irf[i_world+t*nworlds],ssigmaxind_lag,K_sim_irf[i_world+t*nworlds],qmax);
			thrust::for_each(
					begin_hh+t*nhousehold,
					end_hh+t*nhousehold,
					simulateforward(
						d_kopt_ptr,
						d_koptind_ptr,
						d_active_ptr,
						d_k_grid_ptr,
						d_z_grid_ptr,
						d_x_grid_ptr,
						qmax,
						w,
						mmu,
						eps_match.dptr,
						Kind_sim_irf[i_world+t*nworlds],
						i_qmax,
						zind_sim_irf.hvec[i_world+t*nworlds],
						ssigmaxind_lag,
						kind_sim_irf.dptr,
						k_sim_irf.dptr,
						xind_sim_irf.dptr,
						clist.dptr,
						activelist.dptr,
						p,
						irfperiods
							)
							);
			/* printf("simulate forward done\n"); */

			// find aggregate C and active ttheta
			C_sim_irf.hptr[i_world+t*nworlds] = thrust::reduce(clist.dvec.begin(), clist.dvec.end(), (double) 0, thrust::plus<double>())/double(nhousehold);
			int activecount = thrust::reduce(activelist.dvec.begin(), activelist.dvec.end(), (int) 0, thrust::plus<int>());
			double ttheta = double(activecount)/double(nhousehold);
			mmu_sim_irf.hptr[i_world+t*nworlds] = p.aalpha0*pow( max(ttheta,1.0/double(nhousehold)) , -p.aalpha1);
			ttheta_sim_irf.hptr[i_world+t*nworlds] = ttheta;
		/* printf("t=%d \n",t); */
		}
		/* printf("world=%d \n",i_world); */
	}

	// save irf results to be analyze in MATLAB
	save_vec(K_sim_irf,    "./results/K_sim_irf.csv");
	save_vec(Kind_sim_irf, "./results/Kind_sim_irf.csv");
	save_vec(kind_sim_irf, "./results/kind_sim_irf.csv");
	save_vec(k_sim_irf,    "./results/k_sim_irf.csv");
	save_vec(qind_sim_irf, "./results/qind_sim_irf.csv");
	save_vec(q_sim_irf,    "./results/q_sim_irf.csv");
	save_vec(C_sim_irf,    "./results/C_sim_irf.csv");
	save_vec(ttheta_sim_irf,  "./results/ttheta_sim_irf.csv");

	// save ref results
	save_vec(K_sim_ref,    "./results/K_sim_ref.csv");
	save_vec(Kind_sim_ref, "./results/Kind_sim_ref.csv");
	save_vec(kind_sim_ref, "./results/kind_sim_ref.csv");
	save_vec(k_sim_ref,    "./results/k_sim_ref.csv");
	save_vec(qind_sim_ref, "./results/qind_sim_ref.csv");
	save_vec(q_sim_ref,    "./results/q_sim_ref.csv");
	save_vec(C_sim_ref,    "./results/C_sim_ref.csv");
	save_vec(ttheta_sim_ref,  "./results/ttheta_sim_ref.csv");

	// to be safe destroy cuBLAS handle
	cublasDestroy(handle);

	return 0;
}
