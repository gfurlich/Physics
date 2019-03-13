/*-------------------------------------------------------------------------------------//

	Author :        Greg Furlich
	File :          atmosphere.cpp
	Date Created :  2017/09/18
       
	Purpose : 		Calculate the density of the atmoshpere given the altitude and 
					then calculate the interaction depth and slant depth.

	Compiling :		g++ -o atmosphere atmosphere.cpp `root-config --cflags --glibs`

	Execution :		./atmosphere


//-------------------------------------------------------------------------------------*/

//--- Including Library for Compilation ---//

// Atmosphere Header File :
#include "atmosphere.h"

// C Input/Outputs Libraries :
#include <iostream>
#include <sstream>
#include <iomanip>

// Standard C Math Libraries :
#include <cmath>

// Standard ROOT Libraries :
#include <TSystem.h>
#include <TROOT.h>
#include <TClass.h>
#include <TFile.h>
#include <TTree.h>
#include <TF1.h>
#include <TMath.h>

// Using Standard workspace :
using namespace std;

//--- Atmospheric Functions ---//


//atm atmosphere(double h){
/*
Return an class of {temp, pressure, density, depth} at given heights h (m) above sea level. Returns (T,P,rho, X) in K, Pa, kg/m3, g/cm^2. This is the US Standard Atmosphere of 1976.
*/
//}

double atmosphere_temp(double h) {
/*
Returns the pressure at given height h (m) above sea level. Returns T in K. This is the US Standard Atmosphere of 1976.
*/
	int ih ;

	// Determine the Base Atmospheric Level
	for(int i = 0; i<8; i++) {
		if (h < altitudes[i]) {
			ih = i - 1;
			break ;
		}
	}
    double baseTemp = temperatures[ih] ;
    double tempGrad = tempGrads[ih] ;
    double pressureRelBase = pressureRels[ih] ;
    double deltaAltitude = h - altitudes[ih] ;

    double temperature = baseTemp + tempGrad*deltaAltitude ;
    double pressureRel ;

	if( tempGrad == 0) {
    	pressureRel  = pressureRelBase * exp(-gMR*deltaAltitude/1000./baseTemp) ;
	}
	else {
    	pressureRel = pressureRelBase * (pow ((baseTemp/temperature),(gMR/tempGrad/1000.) ) ) ;
	}
    double pressure = pressureRel * pressureSL ;
    double density  = pressureRel * densitySL * temperatureSL/temperature ;

    return temperature ;
}

double atmosphere_pressure(double h) {
/*
Returns the pressure at given height h (m) above sea level. Returns P in Pa. This is the US Standard Atmosphere of 1976.
*/
	int ih ;

	// Determine the Base Atmospheric Level
	for(int i = 0; i<8; i++) {
		if (h < altitudes[i]) {
			ih = i - 1;
			break ;
		}
	}

    double baseTemp = temperatures[ih] ;
    double tempGrad = tempGrads[ih] ;
    double pressureRelBase = pressureRels[ih] ;
    double deltaAltitude = h - altitudes[ih] ;

    double temperature = baseTemp + tempGrad*deltaAltitude ;
    double pressureRel ;

	if( tempGrad == 0) {
    	pressureRel  = pressureRelBase * exp(-gMR*deltaAltitude/1000./baseTemp) ;
	}
	else {
    	pressureRel = pressureRelBase * (pow ((baseTemp/temperature),(gMR/tempGrad/1000.) ) ) ;
	}
    double pressure = pressureRel * pressureSL ;
    double density  = pressureRel * densitySL * temperatureSL/temperature ;

    return pressure ;
}

double atmosphere_rho(double h) {
/*
Return the density at given height h (m) above sea level. Returns rho in kg/m^3. This is the US Standard Atmosphere of 1976.
*/
	int ih ;

	// Determine the Base Atmospheric Level
	for(int i = 0; i<8; i++) {
		if (h < altitudes[i]) {
			ih = i -1 ;
			break ;
		}
	}

    double baseTemp = temperatures[ih] ;
    double tempGrad = tempGrads[ih] ;
    double pressureRelBase = pressureRels[ih] ;
    double deltaAltitude = h - altitudes[ih] ;

    double temperature = baseTemp + tempGrad*deltaAltitude ;
    double pressureRel ;

	if( tempGrad == 0) {
    	pressureRel  = pressureRelBase * exp(-gMR*deltaAltitude/1000./baseTemp) ;
	}
	else {
    	pressureRel = pressureRelBase * (pow ((baseTemp/temperature),(gMR/tempGrad/1000.) ) ) ;
	}
    double pressure = pressureRel * pressureSL ;
    double density  = pressureRel * densitySL * temperatureSL/temperature ;

    return density ;
}

double atmosphere_layer_depth(double h) {
/*
Return the Interaction Depth from the Atmospheric Layer's Base to a given height h (m) above sea level. Returns X in g/cm^2.
*/
	int ih ;

	// Determine the Base Atmospheric Level
	for(int i = 0; i<8; i++) {
		if (h < altitudes[i]) {
			ih = i - 1 ;
			break ;
		}
	}

    double tempBase = temperatures[ih] ;
    double tempGrad = tempGrads[ih] ;
    double pressureBase = pressures[ih];
	double altitudeBase = altitudes[ih] ;
    double deltaAltitude = h - altitudes[ih] ;


    double temperature = tempBase + tempGrad*deltaAltitude ;
	double depth ;

	if( tempGrad == 0) {
    	depth  = (pressureBase / gravity ) * (1 - exp(-gMR*deltaAltitude/1000./tempBase) ) ;
	}
	else {
    	depth = (pressureBase / gravity ) * (1 - pow ((tempBase/temperature),(gMR/tempGrad/1000.) ) ) ;
	}
    depth = depth / 10 ;	// Kg/m^2 -> g/cm^2

	//printf("%i \t %f \t %f \t %f \t %f \t %f \t %f\n", ih, tempBase, tempGrad, temperature, pressureBase,altitudeBase, deltaAltitude);
	//return pressureBase ;
    return depth ;
}

double ROOT_atmo_layer_depth(double h) {
/*
Return the Interaction Depth from the Atmospheric Layer's Base to a given height h (m) above sea level using ROOT Functions. Returns X in g/cm^2.
*/
	int ih ;

	// Determine the Base Atmospheric Level
	for(int i = 0; i<8; i++) {
		if (h < altitudes[i]) {
			ih = i - 1 ;
			break ;
		}
	}

	// Define Atmospheric Layer Parameters :
    double tempBase = temperatures[ih] ;
    double tempGrad = tempGrads[ih] ;
    double pressureBase = pressures[ih];
	double altitudeBase = altitudes[ih] ;
	double altitudeCeiling = altitudes[ih+1] ;
    double deltaAltitude = h - altitudes[ih] ;

    double temperature = tempBase + tempGrad*deltaAltitude ;
	double ROOT_depth ;
	TF1 *root_depth ;

	// Define Functions of Atmoshperic Layer and Set Paramters
	if( tempGrad == 0) {
    	//depth  = (pressureBase / gravity ) * (1 - exp(-gMR*deltaAltitude/1000./tempBase) ) ;
	root_depth  = new TF1("root_depth","([0] / [1] ) * (1 - exp(-[2]*[3]/1000./[4]) )", altitudeBase, altitudeCeiling) ;

	root_depth->SetParameter(0,pressureBase);
	root_depth->SetParameter(1,gravity);
	root_depth->SetParameter(2,gMR);
	root_depth->SetParameter(3,deltaAltitude);
	root_depth->SetParameter(4,tempBase);

	}
	else {
    	// depth = (pressureBase / gravity ) * (1 - pow ((tempBase/temperature),(gMR/tempGrad/1000.) ) ) ;
	root_depth = new TF1("root_depth","([0] / [1] ) * (1 - ( [2] / [3] )**( [4] /[5]/1000. ) )",altitudeBase, altitudeCeiling ) ;

	root_depth->SetParameter(0,pressureBase);
	root_depth->SetParameter(1,gravity);
	root_depth->SetParameter(2,tempBase);
	root_depth->SetParameter(3,temperature);
	root_depth->SetParameter(4,gMR);
	root_depth->SetParameter(5,tempGrad);
	}
    ROOT_depth = root_depth->Eval(h) / 10 ;	// Kg/m^2 -> g/cm^2

	//printf("%i \t %f \t %f \t %f \t %f \t %f \t %f\n", ih, tempBase, tempGrad, temperature, pressureBase,altitudeBase, deltaAltitude);
    return ROOT_depth ;
}

double depth(double h) {
/*
Return the Interaction Depth from the Sea Level to a given height h (m) above sea level. Returns X in g/cm^2. Note zero starts at sea level and not the top of the atmosphere.
*/
	int ih ;
	double depth = 0. ;

	// Determine the Base Atmospheric Level :
	for(int i = 0; i<8; i++) {
		if (h < altitudes[i]) {
			ih = i - 1 ;
			break ;
		}
	}
	
	// Adding the underlying layers :
	//printf("Layer \t  Base Layer \t Altitude (m) \t Depth (g/cm^2) \n");
	for(int i = 1; i <= ih; i++){
		depth = depth + atmosphere_layer_depth(altitudes[i]-.1) ;
		//printf("%i \t %i \t %f \t %f \n", i, ih, altitudes[i]-.1, depth);
	}

	// Adding the layer the altitude lies in:
	depth = depth + atmosphere_layer_depth(h);

	// Make Zero start at the top of the atmosphere:
	// depth = Pb - depth ;

    return depth ;
}

double ROOT_depth(double h) {
/*
Return the Interaction Depth from the Sea Level to a given height h (m) above sea level using ROOT Functions. Returns X in g/cm^2. Note zero starts at sea level and not the top of the atmosphere.
*/
	int ih ;
	double root_depth = 0. ;

	// Determine the Base Atmospheric Level :
	for(int i = 0; i<8; i++) {
		if (h < altitudes[i]) {
			ih = i - 1 ;
			break ;
		}
	}
	
	// Adding the underlying layers :
	//printf("Layer \t  Base Layer \t Altitude (m) \t Depth (g/cm^2) \n");
	for(int i = 1; i <= ih; i++){
		root_depth = root_depth + ROOT_atmo_layer_depth(altitudes[i]-.1) ;
		//printf("%i \t %i \t %f \t %f \n", i, ih, altitudes[i]-.1, depth);
	}

	// Adding the layer the altitude lies in:
	root_depth = root_depth + ROOT_atmo_layer_depth(h);

	// Make Zero start at the top of the atmosphere:
	// depth = Pb - depth ;

    return root_depth ;
}

double slantdepth(double d, double azimuthal) {
/*
Return the Slant Depth from sea level to a given distance (m) at the azimuthal angle (in radians) . Returns X' in g/cm^2. Note zero starts at sea level and not the top of the atmosphere.
*/
	double cq = cos(azimuthal) ;
	double slantdepth = depth(d * cq) / cq ;

    return slantdepth ;
}

double ROOT_slantdepth(double d, double azimuthal) {
/*
Return the Slant Depth from sea level to a given distance (m) at the azimuthal angle (in radians) using ROOT Functions. Returns X' in g/cm^2. Note zero starts at sea level and not the top of the atmosphere.
*/
	double cq = TMath::Cos(azimuthal) ;
	double root_slantdepth = depth(d * cq) / cq ;

    return root_slantdepth ;
}

double height(double X ) {
/*
Return the altitude from sea level given a Interaction Depth. Returns h in meters.
Subtracting the depth, X, from the lower atmospheric layers out is crucial!
*/
	int ih ;

	// Determine the Base Atmospheric Level
	for(int i = 1; i < 8; i++) {
		//printf("Base Depth =  %f \n",depth(altitudes[i]));
		if (X < depth(altitudes[i]) ) {
			ih = i - 1 ;
			break ;
		}
	}

    double tempBase = temperatures[ih] ;
    double tempGrad = tempGrads[ih] ;
    double pressureBase = pressures[ih] ;
	double altitudeBase = altitudes[ih] ;

	double height ;


	// Subtract the Interaction depth of Lower Layers :
	X = X - ROOT_depth(altitudeBase) ;

	// Convert from g/cm^2 to kg/m^2 (SI):
	X = X * 10 ;

	if( tempGrad == 0) {
    	height  = altitudeBase + ( tempBase * 1000. / gMR ) * ( log(  1 / (  1. - ( gravity * X / pressureBase ) ) ) ) ;
		// Having issues with ln(-x)...
	}
	else {
    	height  = altitudeBase + ( tempBase / tempGrad ) * ( pow( (1. / (1. - ( gravity * X / pressureBase ) ) ) , ( tempGrad * 1000. /  gMR) ) - 1 ) ;
	}

	//printf("%i \t %f \t %f \t %f \t %f \t %f \n", ih, tempBase, tempGrad, pressureBase, altitudeBase, pressureBase / gravity);

    return height ;
}

double ROOT_height(double X) {
/*
Return the altitude from sea level given a Interaction Depth. Returns h in meters.
*/

	int ih ;

	// Determine the Base Atmospheric Level
	for(int i = 1; i < 8; i++) {
		//printf("Base Depth =  %f \n",depth(altitudes[i]));
		if (X < depth(altitudes[i]) ) {
			ih = i - 1 ;
			break ;
		}
	}

	// Define Atmospheric Layer Parameters :
    double tempBase = temperatures[ih] ;
    double tempGrad = tempGrads[ih] ;
    double pressureBase = pressures[ih];
	double altitudeBase = altitudes[ih] ;
	double altitudeCeiling = altitudes[ih+1] ;

	double ROOT_height ;
	TF1 *root_depth ;

	// Subtract the Interaction depth of Lower Layers :
	X = X - ROOT_depth(altitudeBase) ;

	// Define Functions of Atmoshperic Layer and Set Paramters
	if( tempGrad == 0) {
    	//depth  = (pressureBase / gravity ) * (1 - exp(-gMR*deltaAltitude/1000./tempBase) ) ;
	root_depth  = new TF1("root_depth","([0] / [1] ) * (1 - exp(-[2]*(x-[3])/1000./[4]) )", altitudeBase, altitudeCeiling) ;

	root_depth->SetParameter(0,pressureBase);
	root_depth->SetParameter(1,gravity);
	root_depth->SetParameter(2,gMR);
	root_depth->SetParameter(3,altitudeBase);
	root_depth->SetParameter(4,tempBase);

	}
	else {
    	// depth = (pressureBase / gravity ) * (1 - pow ((tempBase/temperature),(gMR/tempGrad/1000.) ) ) ;
	root_depth = new TF1("root_depth","([0] / [1] ) * (1 - ( [2] / ([2] + [5]*(x - [3]) ) )**( [4] /[5]/1000. ) ) ",altitudeBase, altitudeCeiling ) ;

	root_depth->SetParameter(0,pressureBase);
	root_depth->SetParameter(1,gravity);
	root_depth->SetParameter(2,tempBase);
	root_depth->SetParameter(3,altitudeBase);
	root_depth->SetParameter(4,gMR);
	root_depth->SetParameter(5,tempGrad);
	}

	X = X * 10 ; // g/cm^2 -> Kg/m^2 
    ROOT_height = root_depth->GetX(X) ;	// Kg/m^2 -> g/cm^2

	//printf("%i \t %f \t %f \t %f \t %f \t %f \t %f \n", ih, tempBase, tempGrad, pressureBase,altitudeBase, altitudeCeiling, 1 - exp(-gMR*(X - altitudeBase)/1000./tempBase));

    return ROOT_height ;
}

double slantheight(double X, double zenith) {
/*
Returns the Distance in (m) above Sea Level from Slant Depth from sea level to a given distance (m) at the zenith angle (in radians) . Note zero starts at sea level and not the top of the atmosphere.
*/
	double cq = cos(zenith) ;
	double distance = height(X * cq) / cq ;

    return distance ;
}

double ROOT_slantheight(double X, double zenith) {
/*
Returns the Distance in (m) above Sea Level from Slant Depth from sea level to a given distance (m) at the zenith angle (in radians) . Note zero starts at sea level and not the top of the atmosphere.
*/
	double cq = TMath::Cos(zenith) ;
	double distance = height(X * cq) / cq ;

    return distance ;
}

double azimuthal(double ux, double uy) {
/*
Returns the azimuthal angle in radians from the shower core unit vectors.
*/
	double azimuthal = atan( ux / uy ) ;

	return azimuthal ;
}

double zenith(double ux, double uy, double uz){
/*
Returns the zenith angle in radians from the shower core unit vectors.
*/
	 double zenith = atan( pow(pow(ux,2) + pow(uy,2),.5) / uz ) ;

	return zenith ;
}

double distance(double xi, double yi, double zi, double xf, double yf, double zf){
/*
Returns the radial distance (m) between 2 points. Use to caluclate distance between FD site and Xmax position.
*/
	double distance = pow( (pow((xf-xi),2) + pow((yf-yi),2) + pow((zf-zi),2)), .5 ) ;

	//printf(" %f %f %f %f %f %f",xf-xi,pow((xf-xi),2),yf-yi,pow((yf-yi),2),zf-zi,pow((zf-zi),2)) ;
 	//printf(" %f\n", pow((xf-xi),2) + pow((yf-yi),2) + pow((zf-zi),2) );
	//printf(" %f\n",distance);
	return distance ;
}

//--- End of Script ---//

