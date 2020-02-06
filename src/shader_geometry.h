//MIT License
//Copyright (c) 2019 Gonzalo G Campos


// Source code for geometry shader
static const GLchar* geometry_shader_source[] =
{
"#version 450 core                                                      \n"
"layout (points) in;                                                    \n"
"layout (triangle_strip) out;                                                   \n"
"layout (max_vertices = 4) out;                                         \n"
"in Vertex                                                              \n"
"{                                                                      \n"
"    float VertexLifeTime;                                              \n"
"    float VertexLifeRemaining;                                         \n"
"}vertex[];                                                             \n"
"void main()                                                            \n"
"{                                                                      \n"
"   vec4 P = gl_in[0].gl_Position;                                      \n"
"   if(vertex[0].VertexLifeRemaining>0)                                 \n"
"   {                                                                   \n"
"       vec2 va = P.xy + vec2(-0.5, 0.5) * 1;                          \n"
"       gl_Position = vec4(va, P.zw);                                   \n"
"       EmitVertex();                                                   \n"
"       va = P.xy + vec2(-0.5, 0.5) * .1;                           \n"
"       gl_Position = vec4(va, P.zw);                                   \n"
"       EmitVertex();                                                   \n"
"       va = P.xy + vec2(0.5, -0.5) * .1;                           \n"
"       gl_Position = vec4(va, P.zw);                                   \n"
"       EmitVertex();                                                   \n"
"       va = P.xy + vec2(0.5, 0.5) * .1;                            \n"
"       gl_Position = vec4(va, P.zw);                                   \n"
"       EmitVertex();                                                   \n"
"   }                                                                   \n"
"   EndPrimitive();                                                     \n"
"}"
};