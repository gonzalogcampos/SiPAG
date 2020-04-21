//MIT License
//Copyright (c) 2019 Gonzalo G Campos

/* PRAGMA */
#pragma once


// Source code for geometry shader
static const GLchar* dots_geometry[] =
{
"#version 450 core                                                      \n"
"layout (points) in;                                                    \n"
"layout (points) out;                                                   \n"
"layout (max_vertices = 1) out;                                         \n"
"in Vertex                                                              \n"
"{                                                                      \n"
"    float VertexLifeTime;                                              \n"
"    float VertexLifeRemaining;                                         \n"
"}vertex[];                                                             \n"
"void main()                                                            \n"
"{                                                                      \n"
"   if(vertex[0].VertexLifeRemaining>0)                                 \n"
"   {                                                                   \n"
"       gl_Position = gl_in[0].gl_Position;                             \n"
"       EmitVertex();                                                   \n"
"   }                                                                   \n"
"   EndPrimitive();                                                     \n"
"}"
};