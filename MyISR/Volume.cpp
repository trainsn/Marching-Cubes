#include <vector>

#include "Volume.h"
#include "GridPoint.h"

using namespace std;

Volume::Volume(int w, int h, int d)
{
	this->xSize = w;
	this->ySize = h;
	this->zSize = d;

	this->max_dim = w;
	this->max_dim = h > this->max_dim ? h : max_dim;
	max_dim = d > max_dim ? d : max_dim;

	this->xSize_norm = this->xSize / (float)max_dim;
	this->ySize_norm = this->ySize / (float)max_dim;
	this->zSize_norm = this->zSize / (float)max_dim;

	this->grids.resize(d);
	for (int i = 0; i < d; i++) {
		this->grids[i].resize(h);
		for (int j = 0; j < h; j++) {
			this->grids[i][j].resize(w);
		}
	}
}
