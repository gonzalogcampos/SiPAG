//MIT License
//Copyright (c) 2019 Gonzalo G Campos

#include <Render.h>
#include <fstream>
#include <sstream>
#include <Console.h>
#include <CudaControler.h>
#include <Values.h>
#include <shaders/default_fragment.h>
#include <shaders/default_geometry.h>
#include <shaders/default_vertex.h>
#include <shaders/dots_fragment.h>
#include <shaders/dots_geometry.h>
#include <shaders/dots_vertex.h>
#include <iostream>


//OPENGL ERROR CALLBACK
void GLAPIENTRY
MessageCallback( GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam )
{
    if(type == GL_DEBUG_TYPE_ERROR)
    {
        std::string msg = "Error in OpenGL: " + std::string(message) + "\n";
        cPrint(msg, 1);
    }
}


void Render::start()
{
	GLenum res = glewInit();
    if (res != GLEW_OK)
    {
        cPrint("Error: failed to init OpenGL\n", 1);
    }
    glEnable( GL_DEBUG_OUTPUT );
    glDebugMessageCallback( MessageCallback, 0 );

    //Compile shaders
    compileShaders();

    //Create buffers
    createBuffers();

    //Send buffers to CUDA
    CudaControler::getInstance()->conectBuffers(bufferX, bufferY, bufferZ, bufferVX, bufferVY, bufferVZ, bufferLT, bufferLR);
}

void Render::draw()
{
	// Clear the screen
    const GLfloat color[] = { 0.0f, 0.f, 0.0f, 1.0f };
    glClearBufferfv(GL_COLOR, 0, color);

    // Select rendering program
    switch(values::render_program)
    {
        case 0:
            glUseProgram(default_program);
            break;
        case 1:
            glUseProgram(dots_program);
            break;
        default:
            glUseProgram(default_program);
            break;
    }

    //Enable buffers
    enableAtrib();

    //Draw particles
    glDrawArrays(GL_POINTS, 0, values::e_MaxParticles);

    //Disable buffers
    disableAtrib();
}

void Render::close()
{
    //Delete programs
    glDeleteProgram(dots_program);
    glDeleteProgram(default_program);

    //Delete buffers
    glDeleteBuffers(1, &bufferX);
    glDeleteBuffers(1, &bufferY);
    glDeleteBuffers(1, &bufferZ);
    glDeleteBuffers(1, &bufferVX);
    glDeleteBuffers(1, &bufferVY);
    glDeleteBuffers(1, &bufferVZ);
    glDeleteBuffers(1, &bufferLT);
    glDeleteBuffers(1, &bufferLR);

}


void Render::createBuffers()
{
    size_t bytes;

    if(values::sys_Double)
    {
        bytes = values::e_MaxParticles*sizeof(double);
    }else
    {
        bytes = values::e_MaxParticles*sizeof(float);
    }
    

    glGenBuffers(1, &bufferX);
    glBindBuffer(GL_ARRAY_BUFFER, bufferX);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);


    glGenBuffers(1, &bufferY);
    glBindBuffer(GL_ARRAY_BUFFER, bufferY);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &bufferZ);
    glBindBuffer(GL_ARRAY_BUFFER, bufferZ);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &bufferVX);
    glBindBuffer(GL_ARRAY_BUFFER, bufferVX);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);


    glGenBuffers(1, &bufferVY);
    glBindBuffer(GL_ARRAY_BUFFER, bufferVY);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &bufferVZ);
    glBindBuffer(GL_ARRAY_BUFFER, bufferVZ);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &bufferLT);
    glBindBuffer(GL_ARRAY_BUFFER, bufferLT);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &bufferLR);
    glBindBuffer(GL_ARRAY_BUFFER, bufferLR);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void Render::compileShaders()
{
        GLuint vertex_shader, geometry_shader, fragment_shader;


        // Default program
        vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, default_vertex, NULL);
        glCompileShader(vertex_shader);
        geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry_shader, 1, default_geometry, NULL);
        glCompileShader(geometry_shader);
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, default_fragment, NULL);
        glCompileShader(fragment_shader);
        default_program = glCreateProgram();
        glAttachShader(default_program, vertex_shader);
        glAttachShader(default_program, geometry_shader);
        glAttachShader(default_program, fragment_shader);
        glLinkProgram(default_program);
        glDeleteShader(vertex_shader);
        glDeleteShader(geometry_shader);
        glDeleteShader(fragment_shader);



        // Dots program
        vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, dots_vertex, NULL);
        glCompileShader(vertex_shader);
        geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry_shader, 1, dots_geometry, NULL);
        glCompileShader(geometry_shader);
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, dots_fragment, NULL);
        glCompileShader(fragment_shader);
        dots_program = glCreateProgram();
        glAttachShader(dots_program, vertex_shader);
        glAttachShader(dots_program, geometry_shader);
        glAttachShader(dots_program, fragment_shader);
        glLinkProgram(dots_program);
        glDeleteShader(vertex_shader);
        glDeleteShader(geometry_shader);
        glDeleteShader(fragment_shader);

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




void Render::enableAtrib()
{
    GLenum type = GL_FLOAT;
    if(values::sys_Double)
        type = GL_DOUBLE;
    
    glBindBuffer(GL_ARRAY_BUFFER, bufferX);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 1, type, GL_FALSE, 0, (GLubyte *)NULL);

    glBindBuffer(GL_ARRAY_BUFFER, bufferY);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, type, GL_FALSE, 0, (GLubyte *)NULL);

    glBindBuffer(GL_ARRAY_BUFFER, bufferZ);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, type, GL_FALSE, 0, (GLubyte *)NULL);

    glBindBuffer(GL_ARRAY_BUFFER, bufferLT);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, type, GL_FALSE, 0, (GLubyte *)NULL);

    glBindBuffer(GL_ARRAY_BUFFER, bufferLR);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 1, type, GL_FALSE, 0, (GLubyte *)NULL);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Render::disableAtrib()
{
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}



bool Render::setTexture(char* file)
{

    bool aux = false;
    if(false)
    {
        int sizeX = 512;
        int sizeY = 512;   
        //generate an OpenGL texture object.
        glGenTextures(1, &texture);

        //binding texture in a 2d
        glBindTexture(GL_TEXTURE_2D, texture);

        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,img_data.GetWidth(), img_data.GetHeight(),0,GL_RGBA, GL_UNSIGNED_BYTE, img_data.GetPixelsPtr());
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)sizeX, (GLsizei)sizeY, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        //Set all the parameters of the texture
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);


        //Tell gl to use this texture
        glBindTexture(GL_TEXTURE_2D, texture);

        aux = true;
    }
    return aux;
}