// Source code for vertex shader
static const GLchar* vertex_shader_source[] =
{
"#version 150 core                              \n"
"void main()                                    \n"
"{                                              \n"
"   gl_Position = vec4(0.0 + 0.1*gl_VertexID, 0.0 + 0.1*gl_VertexID, 0.5, 1.0);      \n"
"}"
};