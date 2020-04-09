//MIT License
//Copyright (c) 2019 Gonzalo G Campos


// Source code for fragment shader
static const GLchar* default_fragment[] =
{
"#version 450 core                                                      \n"
"uniform sampler2D image;                                               \n"
"uniform bool RasAlpha;                                                 \n"
"uniform vec3 defaultColor;                                             \n"
"uniform float timeOpacityGrowing;                                      \n"
"uniform float timeOpacityDecreasing;                                   \n"
"uniform float maxOpacity;                                              \n"
"in vec2 Vertex_UV;                                                     \n"
"in float VertexLifeTime;                                               \n"
"in float VertexLifeRemaining;                                          \n"
"out vec4 color;                                                        \n"
"void main()                                                            \n"
"{                                                                      \n"
"   float l = VertexLifeTime/(VertexLifeTime + VertexLifeRemaining);    \n"
"   float o = 1.f;                                                      \n"
"   float drecreasingStart = (1-timeOpacityDecreasing);                 \n"
"   if(l<timeOpacityGrowing)                                            \n"
"   {                                                                   \n"
"       o = maxOpacity*l/timeOpacityGrowing;                            \n"
"   }else if(l>drecreasingStart){                                       \n"
"       o = maxOpacity*(1-((l-drecreasingStart)/timeOpacityDecreasing));\n"
"   }else{                                                              \n"
"       o = maxOpacity;                                                 \n"
"   }                                                                   \n"
"   color = texture(image, Vertex_UV);                                  \n"
"   if(RasAlpha)                                                        \n"
"   {                                                                   \n"
"       color = vec4(defaultColor.x, defaultColor.y, defaultColor.z, color.r*o);                      \n"
"   }else{                                                              \n"
"       color = vec4(color.r, color.g, color.b, color.a*o);             \n"
"    }                                                                  \n"
"}"
};