//MIT License
//Copyright (c) 2019 Gonzalo G Campos


// Source code for vertex shader
static const GLchar* vertex_shader_dost_source[] =
{
"#version 450 core                                                                              \n"
"layout (location = 0) in float VertexPositionX;                                                \n"
"layout (location = 1) in float VertexPositionY;                                                \n"
"layout (location = 2) in float VertexPositionZ;                                                \n"
"layout (location = 3) in float VertexLifeTime;                                                 \n"
"layout (location = 4) in float VertexLifeRemaining;                                            \n"
//"out float gl_Position;                                                                       \n"
//"out float VertexLifeTime;                                                                    \n"
//"out float VertexLifeRemaining                                                                \n"
"void main()                                                                                    \n"
"{                                                                                              \n"
"   gl_Position = vec4(VertexPositionX, VertexPositionY- 0.8, VertexPositionZ - 1, 1.0);        \n"
"}"
};