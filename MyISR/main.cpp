#pragma comment(lib,"glfw3.lib")
#define _USE_MATH_DEFINES
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <GL/glm/glm.hpp>
#include <GL/glm/gtc/matrix_transform.hpp>
#include <GL/glm/gtc/type_ptr.hpp>
#include <GL/glm/gtx/transform2.hpp>

#include "shader.h"
#include "Volume.h"
#include "GridPoint.h"
#include "marching_cubes.h"

#include <iostream>
#include <vector>
#include <stack> 
#include <stdlib.h>
#include <math.h>

using namespace std;

// settings
const unsigned int SCR_WIDTH = 1024;
const unsigned int SCR_HEIGHT = 1024;

const float eps = 0.00001;

// Isosurface data structures for holding the vertices, vertex normals, and vertex colors.
vector<float> vertices;
vector<float> vertex_normals; 
vector<float> vertex_colors;
vector<float> isosurface_vertices;
vector<float> isosurface_vertex_normals;
vector<float> isosurface_vertex_colors;

// volume information
Volume vol = Volume();

unsigned int vao[1];
unsigned int gbo[3];
unsigned int isosurfaceVertexPositionBuffer;
unsigned int isosurfaceVertexNormalBuffer;
unsigned int isosurfaceVertexColorBuffer;

// A stack for preserving matrix transformation states.
stack<glm::mat4> mvMatrixStack;

glm::mat4 pMatrix;
glm::mat4 mvMatrix;
glm::mat3 normalMatrix;

// Perspective or orthographic projection?
bool perspective_projection = true;

// Base color used for the ambient, fog, and clear-to colors.
glm::vec3 base_color(10.0 / 255.0, 10.0 / 255.0, 10.0 / 255.0);

// Lighting power.
float lighting_power = 2;

// Used for time based animation.
float time_last = 0; 

// Used to rotate the isosurface.
float rotation_radians = 0.0;
float rotation_radians_step = 0.3 * 180 / M_PI;

// Use lighting?
int use_lighting = 1;

// Render different buffers.
int show_depth = 0;
int show_normals = 0;
int show_position = 0;

// Used to orbit the point lights.
float point_light_theta = 1.57;
float point_light_phi = 1.57;
float point_light_theta_step = 0.39;
float point_light_phi_step = 0.39;

float point_light_theta1 = 1.57;
float point_light_phi1 = 1.57;
float point_light_theta_step1 = 0.3;
float point_light_phi_step1 = 0.3;

unsigned int initVol(const char *filename, unsigned int w, unsigned int h, unsigned int d){
	FILE *fp;
	size_t size = w * h*d;
	uint8_t *data = new uint8_t[size];	// 8 bit
	if (!(fp = fopen(filename, "rb"))) {
		cout << "Error: opening .raw file failed" << endl;
		exit(EXIT_FAILURE);
	}
	else {
		cout << "OK: open .raw file succeed" << endl;
	}

	if (fread(data, sizeof(char), size, fp) != size) {
		cout << "Error: read .raw failed" << endl;
	} 
	else {
		cout << "OK: read .raw file succeed" << endl;
	}
	fclose(fp);

	// Generate the cube grid scalar field.
	/*

	j
	|
	|          k
	|         .
	|       .
	|     .
	|    .
	|  .
	|._________________i

	*/
	
	vol = Volume(w, h, d);
	for (int k = 0; k < d; k++) {
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				int idx = k * h * w + j * w + i;
				float z = k / (float)vol.max_dim;
				float y = j / (float)vol.max_dim;
				float x = i / (float)vol.max_dim;
				float value = (float)data[idx];
				vol.grids[k][j][i] = GridPoint(x, y, z, value);					
			}
		}
	}

	for (int k = 0; k < d; k++) {
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				if (k == 0) {
					vol.grids[k][j][i].normal_z = (vol.grids[k + 1][j][i].value - vol.grids[k][j][i].value);
				}
				else if (k == d - 1) {
					vol.grids[k][j][i].normal_z = (vol.grids[k][j][i].value - vol.grids[k - 1][j][i].value);
				}
				else {
					vol.grids[k][j][i].normal_z = 0.5 * (vol.grids[k + 1][j][i].value - vol.grids[k - 1][j][i].value);
				}

				if (j == 0) {
					vol.grids[k][j][i].normal_y = (vol.grids[k][j + 1][i].value - vol.grids[k][j][i].value);
				}
				else if (j == h - 1) {
					vol.grids[k][j][i].normal_y = (vol.grids[k][j][i].value - vol.grids[k][j - 1][i].value);
				}
				else {
					vol.grids[k][j][i].normal_y = 0.5 * (vol.grids[k][j + 1][i].value - vol.grids[k][j - 1][i].value);
				}

				if (i == 0) {
					vol.grids[k][j][i].normal_x = (vol.grids[k][j][i + 1].value - vol.grids[k][j][i].value);
				}
				else if (i == w - 1) {
					vol.grids[k][j][i].normal_x = (vol.grids[k][j][i].value - vol.grids[k][j][i - 1].value);
				}
				else {
					vol.grids[k][j][i].normal_x = 0.5 * (vol.grids[k][j][i + 1].value - vol.grids[k][j][i - 1].value);
				}
			}
		}
	}
}


// Generates one triangle complete with vertex normals and vertex colors.
void triangle(GridPoint p1, GridPoint p2, GridPoint p3, bool invert_normals) {
	// Push the vertices to this triangle face.
	// Pushing point 3, then 2, and then 1 so that the front face of the triangle
	// points outward from the surface.

	// Push point 1, then 2, and then 3 so that the front front face of the triangle
	// points inward from the surface.
	vertices.push_back(p3.x); vertices.push_back(p3.y); vertices.push_back(p3.z);
	vertices.push_back(p2.x); vertices.push_back(p2.y); vertices.push_back(p2.z);
	vertices.push_back(p1.x); vertices.push_back(p1.y); vertices.push_back(p1.z);

	// Calculate the isosurface gradient at point 1, 2, and 3 of the triangle.
	// These three gradient vectors are the vertex normals of this triangle.
	// This will provide a nice smooth appearance when the lighting is calculated.
	// These three gradient vectors will also be the vertex colors.	

	int invert_normal = 1;
	if (invert_normals)
		invert_normal = -1;

	// Point 3
	float vertex_normal_x = p3.normal_x;
	float vertex_normal_y = p3.normal_y;
	float vertex_normal_z = p3.normal_z;

	float vertex_normal_length = sqrt((vertex_normal_x * vertex_normal_x) + (vertex_normal_y * vertex_normal_y) + (vertex_normal_z * vertex_normal_z));
	
	if (vertex_normal_length != 0) {
		vertex_normal_x = vertex_normal_x / vertex_normal_length;
		vertex_normal_y = vertex_normal_y / vertex_normal_length;
		vertex_normal_z = vertex_normal_z / vertex_normal_length;
	}

	vertex_normals.push_back(invert_normal * vertex_normal_x);
	vertex_normals.push_back(invert_normal * vertex_normal_y);
	vertex_normals.push_back(invert_normal * vertex_normal_z);

	// Push the vertex colors for this triangle face point.
	vertex_colors.push_back(1.0);
	vertex_colors.push_back(1.0);
	vertex_colors.push_back(0.0);

	// Point 2
	vertex_normal_x = p2.normal_x;
	vertex_normal_y = p2.normal_y;
	vertex_normal_z = p2.normal_z;

	vertex_normal_length = sqrt((vertex_normal_x * vertex_normal_x) + (vertex_normal_y * vertex_normal_y) + (vertex_normal_z * vertex_normal_z));

	if (vertex_normal_length != 0) {
		vertex_normal_x = vertex_normal_x / vertex_normal_length;
		vertex_normal_y = vertex_normal_y / vertex_normal_length;
		vertex_normal_z = vertex_normal_z / vertex_normal_length;
	}

	vertex_normals.push_back(invert_normal * vertex_normal_x);
	vertex_normals.push_back(invert_normal * vertex_normal_y);
	vertex_normals.push_back(invert_normal * vertex_normal_z);

	// Push the vertex colors for this triangle face point.
	vertex_colors.push_back(1.0);
	vertex_colors.push_back(1.0);
	vertex_colors.push_back(0.0);

	// Point 1
	vertex_normal_x = p1.normal_x;
	vertex_normal_y = p1.normal_y;
	vertex_normal_z = p1.normal_z;

	vertex_normal_length = sqrt((vertex_normal_x * vertex_normal_x) + (vertex_normal_y * vertex_normal_y) + (vertex_normal_z * vertex_normal_z));

	if (vertex_normal_length != 0) {
		vertex_normal_x = vertex_normal_x / vertex_normal_length;
		vertex_normal_y = vertex_normal_y / vertex_normal_length;
		vertex_normal_z = vertex_normal_z / vertex_normal_length;
	}

	vertex_normals.push_back(invert_normal * vertex_normal_x);
	vertex_normals.push_back(invert_normal * vertex_normal_y);
	vertex_normals.push_back(invert_normal * vertex_normal_z);

	// Push the vertex colors for this triangle face point.
	vertex_colors.push_back(1.0);
	vertex_colors.push_back(1.0);
	vertex_colors.push_back(0.0);
}

GridPoint edge_intersection_interpolation(GridPoint cube_va, GridPoint cube_vb, float iso_value) {
	if (abs(iso_value - cube_va.value) < eps)
		return cube_va;
	if (abs(iso_value - cube_vb.value) < eps)
		return cube_vb;
	if (abs(cube_vb.value - cube_va.value) < eps)
		return cube_va;

	float mean = (iso_value - cube_va.value) / (cube_vb.value - cube_va.value);
	float x = cube_va.x + mean * (cube_vb.x - cube_va.x);
	float y = cube_va.y + mean * (cube_vb.y - cube_va.y);
	float z = cube_va.z + mean * (cube_vb.z - cube_va.z);

	float normal_x = cube_va.normal_x + mean * (cube_vb.normal_x - cube_va.normal_x);
	float normal_y = cube_va.normal_y + mean * (cube_vb.normal_y - cube_va.normal_y);
	float normal_z = cube_va.normal_z + mean * (cube_vb.normal_z - cube_vb.normal_z);

	return GridPoint(x, y, z, iso_value, normal_x, normal_y, normal_z);
}

// The marching cubes algorithm.
void marching_cubes(int w, int h, int d, float iso_level, bool invert_normals) {
	for (int k = 0; k < d - 1; k++) {
		for (int j = 0; j < h - 1; j++) {
			for (int i = 0; i < w - 1; i++) {
				// Perform the algorithm on one cube in the grid.

				// The cube's vertices.
				// There are eight of them.

				//    4---------5
				//   /|        /|
				//  / |       / |
				// 7---------6  |
				// |  |      |  |
				// |  0------|--1
				// | /       | /
				// |/        |/
				// 3---------2
				GridPoint cube_v3 = vol.grids[k][j][i]; // Lower left  front corner.
				GridPoint cube_v2 = vol.grids[k][j][i + 1]; // Lower right front corner.
				GridPoint cube_v6 = vol.grids[k][j + 1][i + 1]; // Upper right front corner.
				GridPoint cube_v7 = vol.grids[k][j + 1][i]; // Upper left  front corner.

				GridPoint cube_v0 = vol.grids[k + 1][j][i]; // Lower left  back corner.
				GridPoint cube_v1 = vol.grids[k + 1][j][i + 1]; // Lower right back corner.
				GridPoint cube_v5 = vol.grids[k + 1][j + 1][i + 1]; // Upper right back corner.
				GridPoint cube_v4 = vol.grids[k + 1][j + 1][i]; // Upper left  back corner.

				int cube_index = 0;

				if (cube_v0.value < iso_level) cube_index |= 1;
				if (cube_v1.value < iso_level) cube_index |= 2;
				if (cube_v2.value < iso_level) cube_index |= 4;
				if (cube_v3.value < iso_level) cube_index |= 8;
				if (cube_v4.value < iso_level) cube_index |= 16;
				if (cube_v5.value < iso_level) cube_index |= 32;
				if (cube_v6.value < iso_level) cube_index |= 64;
				if (cube_v7.value < iso_level) cube_index |= 128;

				// Does the isosurface not intersect any edges of the cube?

				if (marching_cubes_edge_table[cube_index] == 0)
					continue;

				// What edges of the cube does the isosurface intersect?
				// For each cube edge intersected, interpolate an intersection vertex between the edge's incident vertices.
				// These vertices of intersection will form the triangle(s) that approximate the isosurface.

				// There are 12 edges in a cube.

				//       4----5----5
				//    8 /|       6/|
				//     / |9      / | 10
				//    7----7----6  |
				//    |  |      |  |
				// 12 |  0---1--|--1
				//    | /       | / 
				//    |/ 4   11 |/ 2
				//    3----3----2
				//
				// 1={0,1},  2={1,2},  3={2,3},  4={3,0},
				// 5={4,5},  6={5,6},  7={6,7},  8={7,4},
				// 9={0,4}, 10={5,1}, 11={2,6}, 12={3,7}

				// Base ten slot: 2048 | 1024 | 512 | 256 | 128 | 64 | 32 | 16 | 8 | 4 | 2 | 1
				// Base two slot:    0 |    0 |   0 |   0 |   0 |  0 |  0 |  0 | 0 | 0 | 0 | 0
				// Edge slot:       12 |   11 |  10 |   9 |   8 |  7 |  6 |  5 | 4 | 3 | 2 | 1  

				vector<GridPoint> vertices_of_intersection(12, GridPoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

				if (marching_cubes_edge_table[cube_index] & 1) // Intersects edge one.
				{
					vertices_of_intersection[0] = edge_intersection_interpolation(cube_v0, cube_v1, iso_level);
				}

				if (marching_cubes_edge_table[cube_index] & 2) // Intersects edge two.
				{
					vertices_of_intersection[1] = edge_intersection_interpolation(cube_v1, cube_v2, iso_level);
				}

				if (marching_cubes_edge_table[cube_index] & 4) // Intersects edge three.
				{
					vertices_of_intersection[2] = edge_intersection_interpolation(cube_v2, cube_v3, iso_level);
				}

				if (marching_cubes_edge_table[cube_index] & 8) // Intersects edge four.
				{
					vertices_of_intersection[3] = edge_intersection_interpolation(cube_v3, cube_v0, iso_level);
				}

				if (marching_cubes_edge_table[cube_index] & 16) // Intersects edge five.
				{
					vertices_of_intersection[4] = edge_intersection_interpolation(cube_v4, cube_v5, iso_level);
				}

				if (marching_cubes_edge_table[cube_index] & 32) // Intersects edge six.
				{
					vertices_of_intersection[5] = edge_intersection_interpolation(cube_v5, cube_v6, iso_level);
				}

				if (marching_cubes_edge_table[cube_index] & 64) // Intersects edge seven.
				{
					vertices_of_intersection[6] = edge_intersection_interpolation(cube_v6, cube_v7, iso_level);
				}

				if (marching_cubes_edge_table[cube_index] & 128) // Intersects edge eight.
				{
					vertices_of_intersection[7] = edge_intersection_interpolation(cube_v7, cube_v4, iso_level);
				}

				if (marching_cubes_edge_table[cube_index] & 256) // Intersects edge nine.
				{
					vertices_of_intersection[8] = edge_intersection_interpolation(cube_v0, cube_v4, iso_level);
				}

				if (marching_cubes_edge_table[cube_index] & 512) // Intersects edge ten.
				{
					vertices_of_intersection[9] = edge_intersection_interpolation(cube_v1, cube_v5, iso_level);
				}

				if (marching_cubes_edge_table[cube_index] & 1024) // Intersects edge eleven.
				{
					vertices_of_intersection[10] = edge_intersection_interpolation(cube_v2, cube_v6, iso_level);
				}

				if (marching_cubes_edge_table[cube_index] & 2048) // Intersects edge twelve.
				{
					vertices_of_intersection[11] = edge_intersection_interpolation(cube_v3, cube_v7, iso_level);
				}

				// Create the triangles.
				// Three vertices make up a triangle per iteration.
				for (int a = 0; marching_cubes_triangle_table[cube_index][a] != -1; a += 3) {
					GridPoint v1 = vertices_of_intersection[marching_cubes_triangle_table[cube_index][a]];
					GridPoint v2 = vertices_of_intersection[marching_cubes_triangle_table[cube_index][a + 1]];
					GridPoint v3 = vertices_of_intersection[marching_cubes_triangle_table[cube_index][a + 2]];

					triangle(v1, v2, v3, invert_normals);
				}
			}
		}
	}
}

void initBuffers() {
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glGenVertexArrays(1, vao);
	glGenBuffers(3, gbo);

	glBindVertexArray(vao[0]);

	// Begin creating the isosurfaces.
	isosurfaceVertexPositionBuffer = gbo[0];
	isosurfaceVertexNormalBuffer = gbo[1];
	isosurfaceVertexColorBuffer = gbo[2];

	// Grid min, grid max, resolution, iso-level, and invert normals.
	// Do not set the resolution to small.
	marching_cubes(vol.xSize, vol.ySize, vol.zSize, 100.0, false);
	isosurface_vertices.swap(vertices);
	isosurface_vertex_normals.swap(vertex_normals);
	isosurface_vertex_colors.swap(vertex_colors);
	
	// Bind and fill the isosurface vertex positions
	glBindBuffer(GL_ARRAY_BUFFER, isosurfaceVertexPositionBuffer);
	glBufferData(GL_ARRAY_BUFFER, isosurface_vertices.size() * sizeof(float), &isosurface_vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);

	// Bind and fill the isosurface vertex normals.
	glBindBuffer(GL_ARRAY_BUFFER, isosurfaceVertexNormalBuffer);
	glBufferData(GL_ARRAY_BUFFER, isosurface_vertex_normals.size() * sizeof(float), &isosurface_vertex_normals[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);

	// Bind and fill the isosurface vertex colors.
	glBindBuffer(GL_ARRAY_BUFFER, isosurfaceVertexColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, isosurface_vertex_colors.size() * sizeof(float), &isosurface_vertex_colors[0], GL_STATIC_DRAW);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
}

void setMatrixUniforms(Shader ourShader) {
	// Pass the vertex shader the projection matrix and the model-view matrix.
	ourShader.setMat4("uPMatrix", pMatrix);
	ourShader.setMat4("uMVMatrix", mvMatrix);

	// Pass the vertex normal matrix to the shader so it can compute the lighting calculations.
	normalMatrix = glm::transpose(glm::inverse(glm::mat3(mvMatrix)));

	ourShader.setMat3("uNMatrix", normalMatrix);
}

void drawScene(Shader ourShader) {
	float time_now = glfwGetTime();
	if (time_last != 0) {
		float time_delta = (time_now - time_last);
		
		rotation_radians += rotation_radians_step * time_delta;
		//rotation_radians = 90.0;
		
		if (rotation_radians > 360)
			rotation_radians = 0.0;

		if (use_lighting == 1) {
			point_light_theta += point_light_theta_step * time_delta;
			point_light_phi += point_light_phi_step * time_delta;

			if (point_light_theta > (M_PI * 2)) point_light_theta = 0.0;
			if (point_light_phi > (M_PI * 2)) point_light_phi = 0.0;

			point_light_theta1 += point_light_theta_step1 * time_delta;
			point_light_phi1 += point_light_phi_step1 * time_delta;

			if (point_light_theta1 > (M_PI * 2)) point_light_theta1 = 0.0;
			if (point_light_phi1 > (M_PI * 2)) point_light_phi1 = 0.0;
		}
	}

	time_last = time_now;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// create the projection matrix 
	float near = 0.1;
	float far = 400.0;
	float fov_r = 60.0f;

	if (perspective_projection) {
		// Resulting perspective matrix, FOV in radians, aspect ratio, near, and far clipping plane.
		pMatrix = glm::perspective(fov_r, (float)SCR_WIDTH / (float)SCR_HEIGHT, near, far);
		// Let the fragment shader know that perspective projection is being used.
		ourShader.setInt("uPerspectiveProjection", 1);
	}
	else {
		// The goal is to have the object be about the same size in the window
		// during orthographic project as it is during perspective projection.

		float a = (float)SCR_WIDTH / (float)SCR_HEIGHT;
		float h = 2 * (25 * tan(fov_r / 2)); // Window aspect ratio.
		float w = h * a; // Knowing the new window height size, get the new window width size based on the aspect ratio.

		// The canvas' origin is the upper left corner. To the right is the positive x-axis. 
		// Going down is the positive y-axis.

		// Any object at the world origin would appear at the upper left hand corner.
		// Shift the origin to the middle of the screen.

		// Also, invert the y-axis as WebgL's positive y-axis points up while the canvas' positive
		// y-axis points down the screen.

		//           (0,O)------------------------(w,0)
		//               |                        |
		//               |                        |
		//               |                        |
		//           (0,h)------------------------(w,h)
		//
		//  (-(w/2),(h/2))------------------------((w/2),(h/2))
		//               |                        |
		//               |         (0,0)          |
		//               |                        |
		// (-(w/2),-(h/2))------------------------((w/2),-(h/2))

		// Resulting perspective matrix, left, right, bottom, top, near, and far clipping plane.
		pMatrix = glm::ortho(-(w / 2),
			(w / 2),
			-(h / 2),
			(h / 2),
			near,
			far);

		// Let the fragment shader know that orthographic projection is being used.
		ourShader.setInt("uPerspectiveProjection", 0);
	}
	
	ourShader.setInt("uShowDepth", show_depth);
	ourShader.setInt("uShowNormals", show_normals);
	ourShader.setInt("uShowPosition", show_position);

	// Move to the 3D space origin.
	mvMatrix = glm::mat4(1.0f);

	// Disable alpha blending.
	glDisable(GL_BLEND);
	if (use_lighting == 1) {
		// Pass the lighting parameters to the fragment shader.
		// Global ambient color. 
		ourShader.setVec3("uAmbientColor", base_color);

		// Point light 1.
		float point_light_position_x = 0 + 13.5 * cos(point_light_theta) * sin(point_light_phi);
		float point_light_position_y = 0 + 13.5 * sin(point_light_theta) * sin(point_light_phi);
		float point_light_position_z = 0 + 13.5 * cos(point_light_phi);

		ourShader.setVec3("uPointLightingColor", lighting_power, lighting_power, lighting_power);
		ourShader.setVec3("uPointLightingLocation", point_light_position_x, point_light_position_y, point_light_position_z);

		// Point light 2.
		float point_light_position_x1 = 0 + 8.0 * cos(point_light_theta1) * sin(point_light_phi1);
		float point_light_position_y1 = 0 + 8.0 * sin(point_light_theta1) * sin(point_light_phi1);
		float point_light_position_z1 = 0 + 8.0 * cos(point_light_phi1);

		ourShader.setVec3("uPointLightingColor1", lighting_power, lighting_power, lighting_power);
		ourShader.setVec3("uPointLightingLocation1", point_light_position_x1, point_light_position_y1, point_light_position_z1);

		// Turn off lighting for a moment so that the point light isosurface is 
		// bright simulating that the light is emanating from the surface.
		use_lighting = 0;

		ourShader.setInt("uUseLighting", use_lighting);

		use_lighting = 1;
	}

	// transform
	glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 2.0f),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f));
	
	glm::mat4 model = glm::mat4(1.0f);
	model *= glm::rotate(rotation_radians, glm::vec3(0.0f, 1.0f, 0.0f));

	// to make the "head256.raw" i.e. the volume data stand up.
	model *= glm::rotate(90.0f, glm::vec3(1.0f, 0.0f, 0.0f));
	model *= glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f));

	mvMatrix = view * model;

	//// First isosurface.
	//mvMatrix *= glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f));

	//// Move down the negative z-axis by 25 units.
	//mvMatrix *= glm::translate(glm::vec3(0.0f, 0.0f, -25.0f));

	//mvMatrix *= glm::scale(glm::vec3(5.0f, 5.0f, 5.0f));

	//// Rotate the model view matrix thereby rotating the isosurface.
	//mvMatrix *= glm::rotate(rotation_radians, glm::vec3(0.0f, 1.0f, 0.0f));

	setMatrixUniforms(ourShader);
	ourShader.setInt("uUseLighting", use_lighting);

	glBindVertexArray(vao[0]);

	//glDrawArrays(GL_LINES, 0, isosurface_vertices.size() / 3);
	glDrawArrays(GL_TRIANGLES, 0, isosurface_vertices.size() / 3);
	glBindVertexArray(0);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

int main()
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	// build and compile our shader program
	// ------------------------------------
	Shader ourShader("shader.vs", "shader.fs"); // you can name your shader files however you like

	//initVol("head256.raw", 256, 256, 225);
	//initVol("marschner_lobb_41x41x41_uint8.raw", 41, 41, 41);
	initVol("nucleon_41x41x41_uint8.raw", 41, 41, 41);

	initBuffers();

	ourShader.use();

	// render loop
	// -----------
	while (!glfwWindowShouldClose(window))
	{
		// input
		// -----
		processInput(window);

		// render
		glClearColor(base_color[0], base_color[1], base_color[2], 1.0); // Set the WebGL background color.
		glEnable(GL_DEPTH_TEST); // Enable the depth buffer.

		drawScene(ourShader);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}