#define nk 2500
#define nz 23
#define tol 1e-7
#define maxiter 2500
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
#include "rbcmodel.h"


/// This function finds the value of RHS given k', k, z
__host__ __device__
double rhsvalue (state s, int i_z, double kplus, int i_kplus, double* EV, para p) {
	return log(s.z*pow(s.k,p.ttheta)+(1-p.ddelta)*s.k-kplus) + p.bbeta*EV[i_kplus+i_z*nk];
};

// This find the max using binary search and assumes concavity
__host__ __device__
void concavemax(double k, double z, const int left_ind, const int right_ind, const int i_k,const int i_z, double* K, double* EV, int* koptind, double* Vplus, para p) {
	int index = i_k + i_z*nk;

	if (right_ind-left_ind==1) {
		double left_value, right_value;
		left_value = rhsvalue(state(k,z,p),i_z,K[left_ind],left_ind, EV, p);
		right_value = rhsvalue(state(k,z,p),i_z,K[right_ind],right_ind, EV, p);
		if (left_value>right_value) {
			Vplus[index] = left_value;
			koptind[index] = left_ind;
		} else {
			Vplus[index] = right_value;
			koptind[index] = right_ind;
		};
	} else if (right_ind-left_ind==2) {
		double value1 = rhsvalue(state(k,z,p),i_z,K[left_ind],left_ind, EV, p);
		double value2 = rhsvalue(state(k,z,p),i_z,K[left_ind+1],left_ind+1, EV, p);
		double value3 = rhsvalue(state(k,z,p),i_z,K[right_ind],right_ind, EV, p);
		if (value1 < value2) {
			if (value2 < value3) {
				Vplus[index] = value3;
				koptind[index] = right_ind;
			} else {
				Vplus[index] = value2;
				koptind[index] = left_ind+1;
			}
		} else {
			if (value1 < value3) {
				Vplus[index] = value3;
				koptind[index] = right_ind;
			} else {
				Vplus[index] = value1;
				koptind[index] = left_ind;
			}
		}
	} else {
		int ind1 = left_ind; int ind4 = right_ind;
		int ind2, ind3;
		double value1, value2, value3;
		while (ind4 - ind1 > 2) {
			ind2 = (ind1+ind4)/2;
			ind3 = ind2 + 1;
			value2 = rhsvalue(state(k,z,p),i_z,K[ind2],ind2, EV, p);
			value3 = rhsvalue(state(k,z,p),i_z,K[ind3],ind3, EV, p);
			if (value2 < value3) {
				ind1 = ind2;
			} else {
				ind4 = ind3;
			};
		};

		// Now the number of candidates is reduced to three
		value1 = rhsvalue(state(k,z,p),i_z,K[ind1],ind1, EV, p);
		value2 = rhsvalue(state(k,z,p),i_z,K[ind4-1],ind4-1, EV, p);
		value3 = rhsvalue(state(k,z,p),i_z,K[ind4],ind4, EV, p);

		if (value1 < value2) {
			if (value2 < value3) {
				Vplus[index] = value3;
				koptind[index] = ind4;
			} else {
				Vplus[index] = value2;
				koptind[index] = ind4-1;
			}
		} else {
			if (value1 < value3) {
				Vplus[index] = value3;
				koptind[index] = ind4;
			} else {
				Vplus[index] = value1;
				koptind[index] = ind1;
			}
		}
	}
};

// This functor optimal kplus and Vplus
struct kplusVplusopt
{
	// Data member
	double *K, *Z, *EV;
	int *koptind;
    double *Vplus;
	para p;

	// Construct this object, create util from _util, etc.
	__host__ __device__
    kplusVplusopt(double* K_ptr, double* Z_ptr, double* EV_ptr, int* koptind_ptr, double* Vplus_ptr, para _p) {
		K = K_ptr; Z = Z_ptr; EV = EV_ptr;
		koptind = koptind_ptr; Vplus = Vplus_ptr;
		p = _p;
	};

	__host__ __device__
	void operator()(int index) {
		// Perform ind2sub
		int subs[2];
		int size_vec[2];
		size_vec[0] = nk;
		size_vec[1] = nz;
		ind2sub(2,size_vec,index,subs);
		int i_k = subs[0];
		int i_z = subs[1];

		// Find and construct state and control, otherwise they won't update in the for loop
		double k =K[i_k]; double z=Z[i_z];

		// Exploit concavity to update V
		concavemax(k, z, 0, nk-1, i_k, i_z, K, EV, koptind, Vplus, p);

	};
};

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
	// Initialize Parameters
	para p; // in #include "rbcmodel.h"

	// Set Model Parameters
	p.bbeta = 0.9825;
	p.ddelta = 0.025;
	p.ttheta = 0.36;
	p.zbar = 1.0;
	p.rrhozz = 0.9457;
	p.std_epsz = 0.0045*0.0045;
	p.complete(); // complete all implied para, find S-S

	std::cout << std::setprecision(16) << "kss: " << p.kss << std::endl;
	std::cout << std::setprecision(16) << "zss: " << p.zbar << std::endl;
	std::cout << std::setprecision(16) << "tol: " << tol << std::endl;

	// Select Device
	// int num_devices;
	// cudaGetDeviceCount(&num_devices);
	if (argc > 1) {
		int gpu = atoi(argv[1]);
		cudaSetDevice(gpu);
	};
	// Only for cuBLAS
	const double alpha = 1.0;
	const double beta = 0.0;

	// Create all STATE, SHOCK grids here
	h_vec_d h_K(nk);
	h_vec_d h_Z(nz);
    h_vec_d h_logZ(nz);
	h_vec_d h_V(nk*nz,0.0);
	h_vec_d h_Vplus(nk*nz,0);
	h_vec_i h_koptind(nk*nz);
	h_vec_d h_EV(nk*nz,0.0);
	h_vec_d h_P(nz*nz, 0);

    load_vec(h_V,"./results/Vgrid.csv"); // in #include "cuda_helpers.h"

	// Create capital grid
	double minK = 1.0/kwidth*p.kss;
	double maxK = kwidth*p.kss;
	linspace(minK,maxK,nk,thrust::raw_pointer_cast(h_K.data())); // in #include "cuda_helpers.h"

	// Create shocks grids
	h_vec_d h_shockgrids(nz);
	double* h_logZ_ptr = thrust::raw_pointer_cast(h_logZ.data());
	double* h_P_ptr = thrust::raw_pointer_cast(h_P.data());
    	tauchen(p.rrhozz, p.std_epsz, h_logZ_ptr, h_P_ptr); // in #include "cuda_helpers.h"
	for (int i_shock = 0; i_shock < nz; i_shock++) {
		h_Z[i_shock] = p.zbar*exp(h_logZ[i_shock]);
	};

	// Copy to the device
	d_vec_d d_K = h_K;
	d_vec_d d_Z = h_Z;
	d_vec_d d_V = h_V;
	d_vec_d d_Vplus = h_Vplus;
	d_vec_i d_koptind = h_koptind;
	d_vec_d d_EV = h_EV;
	d_vec_d d_P = h_P;

	// Obtain device pointers to be used by cuBLAS
	double* d_K_ptr = raw_pointer_cast(d_K.data());
	double* d_Z_ptr = raw_pointer_cast(d_Z.data());
	double* d_V_ptr = raw_pointer_cast(d_V.data());
	double* d_Vplus_ptr = raw_pointer_cast(d_Vplus.data());
	int* d_koptind_ptr = raw_pointer_cast(d_koptind.data());
	double* d_EV_ptr = raw_pointer_cast(d_EV.data());
	double* d_P_ptr = raw_pointer_cast(d_P.data());

	// Firstly a virtual index array from 0 to nk*nk*nz
	thrust::counting_iterator<int> begin(0);
	thrust::counting_iterator<int> end(nk*nz);

    // Create Timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    // Start Timer
	cudaEventRecord(start,NULL);

	// Step.1 Has to start with this command to create a handle
	cublasHandle_t handle;

	// Step.2 Initialize a cuBLAS context using Create function,
	// and has to be destroyed later
	cublasCreate(&handle);

	double diff = 10;  int iter = 0;
	while ((diff>tol)&&(iter<maxiter)){
		// Find EMs for low and high
		cublasDgemm(handle,
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			nk, nz, nz,
			&alpha,
			d_V_ptr,
			nk,
			d_P_ptr,
			nz,
			&beta,
			d_EV_ptr,
			nk);

		// Directly find the new Value function
		thrust::for_each(
			begin,
			end,
			kplusVplusopt(d_K_ptr, d_Z_ptr, d_EV_ptr, d_koptind_ptr, d_Vplus_ptr, p)
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

	//==========cuBLAS stuff ends=======================
	// Step.3 Destroy the handle.
	cublasDestroy(handle);

	// Stop Timer
	cudaEventRecord(stop,NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0;
	cudaEventElapsedTime(&msecTotal, start, stop);

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal;
	std::cout << "Time= " << msecPerMatrixMul << " msec, iter= " << iter << std::endl;

	// Copy back to host and print to file
	h_V = d_V;
	h_EV = d_EV;
	h_koptind = d_koptind;

    save_vec(h_K,"./results/Kgrid.csv"); // in #include "cuda_helpers.h"
	save_vec(h_Z,"./results/Zgrid.csv"); // in #include "cuda_helpers.h"
	save_vec(h_P,"./results/Pgrid.csv"); // in #include "cuda_helpers.h"
	save_vec(h_V,"./results/Vgrid.csv"); // in #include "cuda_helpers.h"
	std::cout << "Policy functions output completed." << std::endl;

	// Export parameters to MATLAB
	p.exportmatlab("./MATLAB/vfi_para.m");

	return 0;
}
