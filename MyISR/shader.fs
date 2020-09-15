#version 330 core
out vec4 FragColor;

uniform int uUseLighting;
uniform int uShowDepth;
uniform int uShowNormals;
uniform int uShowPosition;
uniform int uPerspectiveProjection;

uniform vec3 uAmbientColor;
uniform vec3 uPointLightingLocation;
uniform vec3 uPointLightingColor;
uniform vec3 uPointLightingLocation1;
uniform vec3 uPointLightingColor1;

in vec4 vPosition;
in vec3 vTransformedNormal;
in vec4 vDiffuseColor;

void main()
{
	vec3 ambient = vDiffuseColor.rgb * uAmbientColor;
	vec3 color = ambient;

	if (uUseLighting){
		vec3 light_direction = normalize(uPointLightingLocation - vPosition.xyz);
		vec3 light_direction1 = normalize(uPointLightingLocation1 - vPosition.xyz);
		vec3 eye_direction = -normalize(vPosition.xyz);
		vec3 surface_normal = normalize(vTransformedNormal);

		vec3 diffuse = vDiffuseColor.xyz * uPointLightingColor * max(dot(surface_normal, light_direction), 0.0);
		vec3 diffuse1 = vDiffuseColor.xyz * uPointLightingColor1 * max(dot(surface_normal, light_direction1), 0.0);
		vec3 specular  = uPointLightingColor  * pow( max( dot( reflect( -light_direction,  surface_normal ), eye_direction ), 0.0 ), 100.0 );
		vec3 specular1 = uPointLightingColor1 * pow( max( dot( reflect( -light_direction1, surface_normal ), eye_direction ), 0.0 ), 100.0 );
		float light_outer_radius  = 20.0;
		float light_inner_radius  = 0.0;
		float light_outer_radius1 = 10.0;
		float light_inner_radius1 = 0.0;
		float light_distance  = length( vPosition.xyz - uPointLightingLocation  );
		float light_distance1 = length( vPosition.xyz - uPointLightingLocation1 );
		float attenuation     = 1.0 - smoothstep( light_inner_radius,  light_outer_radius,  light_distance  );
		float attenuation1    = 1.0 - smoothstep( light_inner_radius1, light_outer_radius1, light_distance1 );
		diffuse   = attenuation  * diffuse;
		diffuse1  = attenuation1 * diffuse1;
		specular  = attenuation  * specular;
		specular1 = attenuation1 * specular1;
		color = ambient + diffuse + diffuse1 + specular + specular1;
	}
	vec4 final_color = vec4(color, 1.0);

	float far          = 50.0;
	float fog_coord;
	if ( uPerspectiveProjection ) {
	     fog_coord     = ( gl_FragCoord.z / gl_FragCoord.w ) / far;
	}
	else {
	     fog_coord     = ( gl_FragCoord.z / gl_FragCoord.w );
	}

	FragColor = final_color;

	if ( uShowDepth ) {
		FragColor = mix( vec4( 1.0 ), vec4( vec3( 0.0 ), 1.0 ), smoothstep( 0.1, 1.0, fog_coord ) );
	}
	if (uShowNormals) {
		vec3 nTN      = normalize( vTransformedNormal );
		FragColor  = vec4( nTN.r, nTN.g, nTN.b, 1.0 );
	}
	if (uShowPosition) {
		vec3 nP       = normalize( vPosition.xyz );
		FragColor  = vec4( nP.r, nP.g, nP.b, 1.0 );
	}
}
