#version 150 core

out vec4 Target;
in vec2 v_TexCoord;

uniform sampler2D t_Texture;

void main() {
    Target = texture(t_Texture, v_TexCoord);
}
