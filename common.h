#ifndef COMMON
#define COMMON

// Header file that contains things many files needs to know
// typedef useful things
typedef thrust::host_vector<double> h_vec_d;
typedef thrust::host_vector<int>    h_vec_i;
typedef thrust::device_vector<double> d_vec_d;
typedef thrust::device_vector<int>    d_vec_i;

// A struct that encompasses host and device vectors
template<class T>
struct cudavec {
	// data members contains host and device vectors
	size_t n;
	thrust::host_vector<T> hvec;
	thrust::device_vector<T> dvec;
	T* hptr;
	T* dptr;

	// constructor, with random default value
	cudavec(size_t _n) {
		n = _n;
		hvec = thrust::host_vector<T>(n);
		dvec = thrust::device_vector<T>(n);
		hptr = thrust::raw_pointer_cast(hvec.data());
		dptr = thrust::raw_pointer_cast(dvec.data());
	};

	// constructor, with specified value x
	cudavec(size_t _n, T x) {
		n = _n;
		hvec = thrust::host_vector<T>(n,x);
		dvec = thrust::device_vector<T>(n,x);
		hptr = thrust::raw_pointer_cast(hvec.data());
		dptr = thrust::raw_pointer_cast(dvec.data());
	};

	// get size
	size_t size() {
		return n;
	};

	// copy from host to device
	void h2d() {
		dvec = hvec;
	};

	// copy from device to host
	void d2h() {
		hvec = dvec;
	};

	// on the host, wrap the [] operator, you can directly write
	// the host_vector. pretty bad practice though.
	__host__
	T& operator[] (const int i){
		return hptr[i];
	};
};
#endif
