#include "GridPoint.h"

GridPoint::GridPoint(float x, float y, float z, float value) {
	this->x = x;
	this->y = y;
	this->z = z;
	this->value = value;
}

GridPoint::GridPoint(float x, float y, float z, float value,
	float normal_x, float normal_y, float normal_z) {
	this->x = x;
	this->y = y;
	this->z = z;
	this->value = value;
	this->normal_x = normal_x;
	this->normal_y = normal_y;
	this->normal_z = normal_z;
}
