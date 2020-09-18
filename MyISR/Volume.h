#ifndef VOLUME_H
#define VOLUME_H

#include <vector>

#include "GridPoint.h"

using namespace std;

class Volume {
public:
	int xSize, ySize, zSize; // size of the original volume
	int max_dim; // the max dimension of the original volume 
	float xSize_norm, ySize_norm, zSize_norm; // size of the normalized volume 
	vector<GridPoint> grids;
	Volume() {};
	Volume(int w, int h, int d);
};

#endif