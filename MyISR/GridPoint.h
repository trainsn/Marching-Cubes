#ifndef GRIDPOINT_H
#define GRIDPOINT_H

struct GridPoint
{
public:
	float x, y, z, value, normal_x, normal_y, normal_z;
	GridPoint() {};
	GridPoint(float x, float y, float z, float value);
	GridPoint(float x, float y, float z, float value,
		float normal_x, float normal_y, float normal_z);
};

#endif
