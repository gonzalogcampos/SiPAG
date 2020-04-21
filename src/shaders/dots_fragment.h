//MIT License
//Copyright (c) 2019 Gonzalo G Campos

/* PRAGMA */
#pragma once


// Source code for fragment shader
static const GLchar* dots_fragment[] =
{
"#version 450 core                              \n"
"out vec4 color;                                \n"
"uniform vec4 dotsColor;                        \n"
"void main()                                    \n"
"{                                              \n"
"   color = dotsColor;                          \n"
"}                                              "
};