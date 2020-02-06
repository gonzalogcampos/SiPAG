// Source code for geometry shader
static const GLchar* geometry_shader_source[] =
{
"#version 450 core                                                      \n"
"layout (points) in;                                                    \n"
"layout (triangle_strip) out;                                           \n"
"layout (max_vertices = 4) out;                                         \n"

"out float VertexLifeTime;                                              \n"
"out float VertexLifeRemaining                                          \n"
"void main()                                                            \n"
"{                                                                      \n"
"   gl_Position = vec4(VertexPositionX, VertexPositionY, .5, 1.0); \n"
"}"