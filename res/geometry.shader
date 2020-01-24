#version 450 core

layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;

out vec2 Vertex_UV;
out vec4 Vertex_Color;
   
void main (void)
{
  vec4 P = gl_Position;

  // a: left-bottom 
  vec2 va = P.xy + vec2(-0.5, -0.5) * 1;
  gl_Position = projectionMatrix * vec4(va, P.zw);
  Vertex_UV = vec2(0.0, 0.0);
  Vertex_Color = vertex[0].color;
  EmitVertex();  
  
  // b: left-top
  vec2 vb = P.xy + vec2(-0.5, 0.5) * particle_size;
  gl_Position = gxl3d_ProjectionMatrix * vec4(vb, P.zw);
  Vertex_UV = vec2(0.0, 1.0);
  Vertex_Color = vertex[0].color;
  EmitVertex();  
  
  // d: right-bottom
  vec2 vd = P.xy + vec2(0.5, -0.5) * particle_size;
  gl_Position = gxl3d_ProjectionMatrix * vec4(vd, P.zw);
  Vertex_UV = vec2(1.0, 0.0);
  Vertex_Color = vertex[0].color;
  EmitVertex();  

  // c: right-top
  vec2 vc = P.xy + vec2(0.5, 0.5) * particle_size;
  gl_Position = gxl3d_ProjectionMatrix * vec4(vc, P.zw);
  Vertex_UV = vec2(1.0, 1.0);
  Vertex_Color = vertex[0].color;
  EmitVertex();  

  EndPrimitive();  
} 