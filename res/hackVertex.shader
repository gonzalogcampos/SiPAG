#version 150
in vec4 gxl3d_Position;
in vec4 gxl3d_Color;

// GLSL Hacker automatic uniforms:
uniform mat4 gxl3d_ModelViewMatrix;

out Vertex
{
  vec4 color;
} vertex;

void main()
{
  gl_Position = gxl3d_ModelViewMatrix * gxl3d_Position;
  vertex.color = gxl3d_Color;
}