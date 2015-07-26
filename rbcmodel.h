#include "common.h"

// Define an class that contains parameters and steady states
struct para {
	// Model parameters
	double bbeta ;
	double ddelta;
	double ttheta;
	double zbar  ;
	double rrhozz;
	double std_epsz;

	// Steady States
	double kss;
	double css;
	double yss;


	// Find steady state and find aalpha based steady state target
	__host__ __device__
	void complete() {
		// Find aalpha based on SS computation
		kss = pow((1/bbeta-(1-ddelta))/ttheta,1/(ttheta-1));
        css = pow(kss,ttheta)-ddelta*kss;
	};

	// Export parameters to a .m file in MATLAB syntax
	__host__
	void exportmatlab(std::string filename) {
		std::ofstream fileout(filename.c_str(), std::ofstream::trunc);

		// Model Parameters
		fileout << std::setprecision(16) << "bbeta=" << bbeta << ";"<< std::endl;
		fileout << std::setprecision(16) << "ddelta=" << ddelta << ";"<< std::endl;
		fileout << std::setprecision(16) << "ttheta=" << ttheta << ";"<< std::endl;
		fileout << std::setprecision(16) << "zbar=" << zbar << ";"<< std::endl;
		fileout << std::setprecision(16) << "rrhozz=" << rrhozz << ";"<< std::endl;
		fileout << std::setprecision(16) << "ssigmaepsz=" << std_epsz << ";"<< std::endl;

		// Steady States
		fileout << std::setprecision(16) << "kss=" << kss << ";"<< std::endl;
		fileout << std::setprecision(16) << "css=" << css << ";"<< std::endl;
		fileout.close();
	};
};

// Define state struct that contains "natural" state 
struct state {
	// Data member
	double k, z, y;

	// Constructor
	__host__ __device__
	state(double _k, double _z, para p) {
		k = _k;
		z = _z;
		y = pow(_k,p.ttheta)*_z;
	};
};
