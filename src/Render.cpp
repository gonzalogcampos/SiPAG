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
#include <SOIL.h>

#include <GL/glm/gtc/matrix_transform.hpp>
#include <GL/glm/gtc/type_ptr.hpp>

const float CameraVelocity = 10.f;


float c_Rotation = 0.f;
float c_Distance = 5.f;
float c_Height = 0.f;


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


    setTexture((char*)"res/text.png");
}

void Render::draw(float dt)
{
    time += dt;
    this->dt = dt;

	// Clear the screen
    const GLfloat color[] = { 0.0f, 0.f, 0.0f, 1.0f };
    glClearBufferfv(GL_COLOR, 0, color);

    //Enable buffers
    enableAtrib();

    //Uniform values
    paseUniforms();

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

        defaultVP  = glGetUniformLocation(default_program, "VP");



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


        dotsVP  = glGetUniformLocation(default_program, "VP");

        glUseProgram(default_program);
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



void Render::setTexture(char* file)
{
    texture = SOIL_load_OGL_texture(file, 0, texture, SOIL_FLAG_MULTIPLY_ALPHA);

    if(texture == 0)
        cPrint("Error loading " + cString(file) + "\n", 2);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}


void Render::paseUniforms()
{
    //VP MATRIX
    const float radius = camD;
    float camX = sin(c_Rotation) * c_Distance;
    float camZ = cos(c_Rotation) * c_Distance;

    glm::mat4 V = glm::lookAt(glm::vec3(camX, c_Height, camZ), glm::vec3(0.f, c_Height, 0.f), glm::vec3(0.0, 1.0, 0.0));
    glm::mat4 P = glm::perspective(0.8f, 4.0f / 3.0f, 0.1f, 100.0f);
    glm::mat4 VP = P * V;


    glUniformMatrix4fv( defaultVP, 1, GL_FALSE, glm::value_ptr( VP ) );
    glUniformMatrix4fv( dotsVP, 1, GL_FALSE, glm::value_ptr( VP ) );
}

void Render::changeShader()
{
    if(defaultShader)
    {
        defaultShader = false;
        glUseProgram(dots_program);
    }else
    {
        defaultShader = true;
        glUseProgram(default_program);
    }
    
}