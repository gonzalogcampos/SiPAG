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



    //Creamos los buffers
    createBuffers();
    //Los mandamos a cuda
    //CudaControler::getInstance()->sendBuffer(bufferX);
    

    rendering_program = compileShaders();

    
    glCreateVertexArrays(1, &vertex_array_object);
    glBindVertexArray(vertex_array_object);

}

void Render::draw()
{
	// Clear the screen
    const GLfloat color[] = { 1.0f, 0.f, 0.0f, 1.0f };
    glClearBufferfv(GL_COLOR, 0, color);

    // Use the program object we created earlier for rendering
    glUseProgram(rendering_program);

    // Draw one point
    glPointSize(10.f);
    glDrawArrays(GL_POINTS, 0, 1);
}

void Render::close()
{
    glDeleteProgram(rendering_program);
    glDeleteVertexArrays(1, &vertex_array_object);
}


void Render::createBuffers()
{
    size_t bytes = values::e_MaxParticles*sizeof(float);

    glGenBuffers(1, &bufferX);
    glBindBuffer(GL_ARRAY_BUFFER, bufferX);
    glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);


    // This is the data that we will place into the buffer object
    //static const float data[] = {2.5, 1.0, 5.0, 8.8, 3.8};
    // Get a pointer to the buffer's data store
    //void * ptr = glMapNamedBuffer(bufferX, GL_WRITE_ONLY);
    // Copy our data into it...
    //memcpy(ptr, data, bytes);
    // Tell OpenGL that we're done with the pointer
    //glUnmapNamedBuffer(GL_ARRAY_BUFFER);

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




