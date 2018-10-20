#version 150 core

in vec3 a_Pos;

uniform Transform {
    mat4 u_Transform;
};

void main() {
    vec4 clipped = u_Transform * vec4(a_Pos, 1.);
    vec3 ndc = clipped.xyz / clipped.w;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc.xyz, 1.);
}
