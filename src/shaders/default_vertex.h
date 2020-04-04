//MIT License
//Copyright (c) 2019 Gonzalo G Campos


// Source code for vertex shader
static const GLchar* default_vertex[] =
{
"#version 450 core                                                                              \n"
"layout (location = 0) in float VertexPositionX;                                                \n"
"layout (location = 1) in float VertexPositionY;                                                \n"
"layout (location = 2) in float VertexPositionZ;                                                \n"
"layout (location = 3) in float VertexLifeTime;                                                 \n"
"layout (location = 4) in float VertexLifeRemaining;                                            \n"
"uniform mat4 VP;                                                                               \n"
"out Vertex                                                                                     \n"
"{                                                                                              \n"
"  float VertexLifeTime;                                                                        \n"
"  float VertexLifeRemaining;                                                                   \n"
"}vertex;                                                                                       \n"
"void main()                                                                                    \n"
"{                                                                                              \n"
"   gl_Position = VP * vec4(VertexPositionX, VertexPositionY, VertexPositionZ, 1.0);            \n"
"   vertex.VertexLifeTime = VertexLifeTime;                                                     \n"
"   vertex.VertexLifeRemaining = VertexLifeRemaining;                                           \n"
"}"
};