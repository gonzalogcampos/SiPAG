#include <Render.h>
#include <iostream>
#include <GL/glew.h>
#include <fstream>
#include <sstream>
#include <Console.h>


#include <Values.h>

void Render::start()
{
	GLenum res = glewInit();
    if (res != GLEW_OK)
    {
        cPrint("Error: failed to init OpenGL\n", 1);
    }
    glEnable( GL_DEBUG_OUTPUT );
    //glDebugMessageCallback( (GLDEBUGPROC) MessageCallback, 0 );

	//Enable Z-Buffer 
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS); 
	glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    createBuffers();
    //Esta mandanga va en cuda
    //cudaGraphicsGLRegisterBuffer(&cudaResourceBuf, bufferX, cudaGraphicsRegisterFlagsNone);

    compileShaders();

}

void Render::draw()
{
	// Clear the screen
    glClearColor(1, 0.0, 0.0, 1.0);
  	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Render::close()
{


}


void Render::createBuffers()
{ 
    cPrint("Creating buffer\n", 3);
    size_t bytes = values::e_MaxParticles*sizeof(float);

    GLuint bufferX;


    glGenBuffers(1, &bufferX);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferX);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, bytes, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,0);
    cPrint("Buffer Created!\n", 3);

}


int Render::compileShaders()
{
        GLuint vertex_shader;
        GLuint fragment_shader;
        GLuint program;

        //Load shader sources
        const GLchar* fragment_shader_source = loadShader((char*)"../src/fragment.shader").c_str();
        const GLchar* vertex_shader_source = loadShader((char*)"../src/vertex.shader").c_str();


        // Create and compile vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
        glCompileShader(vertex_shader);
        // Create and compile fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
        glCompileShader(fragment_shader);
        // Create program, attach shaders to it, and link it
        program = glCreateProgram();
        glAttachShader(program, vertex_shader);
        glAttachShader(program, fragment_shader);
        glLinkProgram(program);
        // Delete the shaders as the program has them now
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);

        return program;
}




std::string Render::loadShader(char* path)
{
    std::string line, allLines;
    std::ifstream theFile(path);
    if (theFile.is_open())
    {
        while (std::getline(theFile, line))
        {
            allLines = allLines + line + "\n";
        }
        theFile.close();
    }
    else
    {
        cPrint("Error: Unable to open shader: " + std::string(path) + "\n", 1);
    }

    return allLines;
}