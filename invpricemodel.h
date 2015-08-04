#ifndef MODEL
#define MODEL

// Define an class that contains parameters and steady states
struct para {
	// Model parameters
	double bbeta;           //if you don't know beta... good luck
	double ttau;            //search cost
	double aalpha;          //y = z*x*k^aalpha*l^v
	double v;               //labor share
	double ddelta;          //depreciation
	double pphi;            //price of disinvestment relative to investment
	double MC;              //How many units of consumption goods is needed for 1 inv good
	double rrhox;           //persistence of idio TFP
	double ppsi;            //quadratic cost of investment adjustment
	double rrhoz;           //persistence of agg TFP
	double ssigmaz;         //std of z innov
	double ssigmax_low;     //low std of x innov
	double ssigmax_high;    //high std of x innov
	double ppsi_n;          //labor distutility
	double aalpha0;         //efficient coeff of matching
	double aalpha1;         //elasticity of matching
	double eeta;            //adjustment coefficient
	double Pssigmax [4];    //Transition prob of ssigmax

	// Export parameters to a .m file in MATLAB syntax
	__host__
	void exportmatlab(std::string filename) {
		std::ofstream fileout(filename.c_str(), std::ofstream::trunc);
		fileout << std::setprecision(16) << "bbeta=       " << bbeta          << ";"<< std::endl;
		fileout << std::setprecision(16) << "ttau=        " << ttau           << ";"<< std::endl;
		fileout << std::setprecision(16) << "aalpha=      " << aalpha         << ";"<< std::endl;
		fileout << std::setprecision(16) << "v=           " << v              << ";"<< std::endl;
		fileout << std::setprecision(16) << "ddelta=      " << ddelta         << ";"<< std::endl;
		fileout << std::setprecision(16) << "pphi=        " << pphi           << ";"<< std::endl;
		fileout << std::setprecision(16) << "MC=          " << MC             << ";"<< std::endl;
		fileout << std::setprecision(16) << "rrhox=       " << rrhox          << ";"<< std::endl;
		fileout << std::setprecision(16) << "ppsi=        " << ppsi           << ";"<< std::endl;
		fileout << std::setprecision(16) << "rrhoz=       " << rrhoz          << ";"<< std::endl;
		fileout << std::setprecision(16) << "ssigmaz=     " << ssigmaz        << ";"<< std::endl;
		fileout << std::setprecision(16) << "ssigmax_low= " << ssigmax_low    << ";"<< std::endl;
		fileout << std::setprecision(16) << "ssigmax_high=" << ssigmax_high   << ";"<< std::endl;
		fileout << std::setprecision(16) << "ppsi_n=      " << ppsi_n         << ";"<< std::endl;
		fileout << std::setprecision(16) << "aalpha0=     " << aalpha0        << ";"<< std::endl;
		fileout << std::setprecision(16) << "aalpha1=     " << aalpha1        << ";"<< std::endl;
		fileout.close();
	};
};

// Define state struct that contains "natural" state
struct state {
	// Data member
	double k, z, y;
	int i_s,i_q;
};

// Define struct that contains the coefficients on the agg. rules
struct aggrules {
	// Data member
	double pphi_KC,      pphi_KK,      pphi_Kz,      pphi_Kssigmax;
	double pphi_qC,      pphi_qK,      pphi_qz,      pphi_qssigmax;
	double pphi_CC,      pphi_CK,      pphi_Cz,      pphi_Cssigmax;
	double pphi_tthetaC, pphi_tthetaK, pphi_tthetaz, pphi_tthetassigmax, pphi_tthetaq;

	// savetofile function
	__host__
	void savetofile(std::string filename) {
		std::cout << "================================================================================" << std::endl;
		std::cout << "Saving to " << filename << std::endl;
		std::ofstream fileout(filename.c_str(), std::ofstream::trunc);
		fileout << std::setprecision(16) << pphi_KC << '\n';
		fileout << std::setprecision(16) << pphi_KK << '\n';
		fileout << std::setprecision(16) << pphi_Kz << '\n';
		fileout << std::setprecision(16) << pphi_Kssigmax << '\n';
		fileout << std::setprecision(16) << pphi_qC << '\n';
		fileout << std::setprecision(16) << pphi_qK << '\n';
		fileout << std::setprecision(16) << pphi_qz << '\n';
		fileout << std::setprecision(16) << pphi_qssigmax << '\n';
		fileout << std::setprecision(16) << pphi_CC << '\n';
		fileout << std::setprecision(16) << pphi_CK << '\n';
		fileout << std::setprecision(16) << pphi_Cz << '\n';
		fileout << std::setprecision(16) << pphi_Cssigmax << '\n';
		fileout << std::setprecision(16) << pphi_tthetaC << '\n';
		fileout << std::setprecision(16) << pphi_tthetaK << '\n';
		fileout << std::setprecision(16) << pphi_tthetaz << '\n';
		fileout << std::setprecision(16) << pphi_tthetassigmax << '\n';
		fileout << std::setprecision(16) << pphi_tthetaq << '\n';
		fileout.close();
		std::cout << "Done!" << std::endl;
		std::cout << "================================================================================" << std::endl;
	};

	// load from function
	__host__
	void loadfromfile(std::string filename) {
		std::cout << "================================================================================" << std::endl;
		std::cout << "Loading to " << filename << std::endl;
		std::ifstream filein(filename.c_str());
		filein >> pphi_KC            ;
		filein >> pphi_KK            ;
		filein >> pphi_Kz            ;
		filein >> pphi_Kssigmax      ;
		filein >> pphi_qC            ;
		filein >> pphi_qK            ;
		filein >> pphi_qz            ;
		filein >> pphi_qssigmax      ;
		filein >> pphi_CC            ;
		filein >> pphi_CK            ;
		filein >> pphi_Cz            ;
		filein >> pphi_Cssigmax      ;
		filein >> pphi_tthetaC       ;
		filein >> pphi_tthetaK       ;
		filein >> pphi_tthetaz       ;
		filein >> pphi_tthetassigmax ;
		filein >> pphi_tthetaq       ;
		filein.close();
		std::cout << "Done!" << std::endl;
		std::cout << "================================================================================" << std::endl;
	};
};

#endif
