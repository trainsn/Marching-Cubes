#version 330 core
layout (location = 0) in vec3 aVertexPosition;
layout (location = 1) in vec3 aVertexNormal;
layout (location = 2) in vec4 aVertexColor; 

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat3 uNMatrix;

out vec4 vPosition;
out vec4 vDiffuseColor;
out vec3 vTransformedNormal;

void main()
{
	vDiffuseColor = aVertexColor;
	vTransformedNormal = uNMatrix * aVertexNormal;
	vPosition          = uMVMatrix * vec4( aVertexPosition, 1.0 );
    gl_Position        = uPMatrix * vPosition;
}