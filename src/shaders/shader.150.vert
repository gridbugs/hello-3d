#version 150 core

in vec3 a_Pos;
in vec2 a_TexCoord;

uniform Transform {
    mat4 u_Transform;
};

uniform Properties {
    vec2 u_AtlasDimensions;
};

out vec2 v_TexCoord;

void main() {
    v_TexCoord = a_TexCoord / u_AtlasDimensions;
    gl_Position = u_Transform * vec4(a_Pos, 1.);
}
