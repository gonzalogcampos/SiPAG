//MIT License
//Copyright (c) 2019 Gonzalo G Campos


// Source code for fragment shader
static const GLchar* default_fragment[] =
{
"#version 450 core                                                      \n"
"uniform sampler2D image                                                \n"
"in vec2 Vertex_UV;                                                     \n"
"in float VertexLifeTime;                                               \n"
"in float VertexLifeRemaining;                                          \n"
"out vec4 color;                                                        \n"
"void main()                                                            \n"
"{                                                                      \n"
"   color = texture(image, VertexUV);                                   \n"
"}"
};