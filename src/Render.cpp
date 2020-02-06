#include <Render.h>
#include <fstream>
#include <sstream>
#include <Console.h>
#include <CudaControler.h>
#include <Values.h>
#include <shader_fragment.h>
#include <shader_vertex.h>


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


    rendering_program = compileShaders();


    //Creamos los buffers
    createBuffers();
    //Los mandamos a cuda
    CudaControler::getInstance()->conectBuffers(bufferX, bufferY, bufferZ, bufferLT, bufferLR);
    


    glCreateVertexArrays(1, &vertex_array_object);
    glBindVertexArray(vertex_array_object);

}

void Render::draw()
{
	// Clear the screen
    const GLfloat color[] = { 0.0f, 0.f, 0.0f, 1.0f };
    glClearBufferfv(GL_COLOR, 0, color);

    // Use the program object we created earlier for rendering
    glUseProgram(rendering_program);


    //glVertexAttribPointer(bufferX, 1, GL_FLOAT, GL_FALSE, sizeof(float), &bufferX);


    enableAtrib();

    // Draw one point
    glPointSize(1.f);
    glDrawArrays(GL_POINTS, 0, values::e_MaxParticles);


    diableAtrib();
    //glDrawElements(GL_POINTS, 0, )
}

void Render::close()
{
    glDeleteProgram(rendering_program);
    glDeleteVertexArrays(1, &vertex_array_object);


    glDeleteBuffers(1, &bufferX);
    glDeleteBuffers(1, &bufferY);
    glDeleteBuffers(1, &bufferZ);
    glDeleteBuffers(1, &bufferLT);
    glDeleteBuffers(1, &bufferLR);

}


void Render::createBuffers()
{
    size_t bytes = values::e_MaxParticles*sizeof(float);
    

    glGenBuffers(1, &bufferX);
    glBindBuffer(GL_ARRAY_BUFFER, bufferX);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);


    glGenBuffers(1, &bufferY);
    glBindBuffer(GL_ARRAY_BUFFER, bufferY);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &bufferZ);
    glBindBuffer(GL_ARRAY_BUFFER, bufferZ);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &bufferLT);
    glBindBuffer(GL_ARRAY_BUFFER, bufferLT);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &bufferLR);
    glBindBuffer(GL_ARRAY_BUFFER, bufferLR);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


GLuint Render::compileShaders()
{
        GLuint vertex_shader;
        GLuint fragment_shader;
        GLuint program;

        //Load shader sources
        //const GLchar* fragment_shader_source = loadShader((char*)"res/fragment.shader").c_str();
        //const GLchar* vertex_shader_source = loadShader((char*)"res/vertex.shader").c_str();

        // Create and compile vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, vertex_shader_source, NULL);
        glCompileShader(vertex_shader);
        // Create and compile fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, fragment_shader_source, NULL);
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




void Render::enableAtrib()
{
    glBindBuffer(GL_ARRAY_BUFFER, bufferX);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);

    glBindBuffer(GL_ARRAY_BUFFER, bufferY);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);

    glBindBuffer(GL_ARRAY_BUFFER, bufferZ);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);

    glBindBuffer(GL_ARRAY_BUFFER, bufferLT);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);

    glBindBuffer(GL_ARRAY_BUFFER, bufferLR);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 0, (GLubyte *)NULL);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Render::diableAtrib()
{
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}