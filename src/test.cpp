#include <GL/glew.h>

#include <GL/freeglut.h>

#include <iostream>
#include <fstream>
#include <sstream>

void step(void), close();
GLuint compileShaders(void);
std::string loadShader(char* path);

int window;
GLuint vertex_array_object;
GLuint rendering_program;

int main(int argv, char **argc) {

    glutInit(&argv, argc);
    glutInitWindowSize(720, 720);
    glutInitDisplayMode(GLUT_RGB | GLUT_STENCIL | GLUT_DOUBLE | GLUT_DEPTH);

    window = glutCreateWindow("Test render");



            //OPEN GL
            if (glewInit() != GLEW_OK)std::cout<<"Error: failed to init OpenGL\n";
            
            //glEnable( GL_DEBUG_OUTPUT );

            //Enable Z-Buffer 
            //glEnable(GL_DEPTH_TEST);
            //glDepthFunc(GL_LESS); 
            //glEnable(GL_CULL_FACE);
            //glCullFace(GL_BACK);

            rendering_program = compileShaders();

            glCreateVertexArrays(1, &vertex_array_object);
            glBindVertexArray(vertex_array_object);


    glutDisplayFunc(step);


    glutMainLoop();

    std::cout<<"Closing\n";

    close();
}

void close()
{
    glDeleteVertexArrays(1, &vertex_array_object);
    glDeleteProgram(rendering_program);
    glDeleteVertexArrays(1, &vertex_array_object);
}

void step(void)
{
    // Clear the screen
    glClearColor(1, 0.0, 0.0, 1.0);
  	glClearBufferfv(GL_COLOR, 0, color);

    // Use the program object we created earlier for rendering
    glUseProgram(rendering_program);
    // Draw one point
    glPointSize(40.0);
    glDrawArrays(GL_POINTS, 0, 1);

    glutSwapBuffers();
    glutPostRedisplay();
}




GLuint compileShaders(void)
{
        GLuint vertex_shader;
        GLuint fragment_shader;
        GLuint program;

        //Load shader sources
        const GLchar* fragment_shader_source = loadShader((char*)"res/fragment.shader").c_str();
        const GLchar* vertex_shader_source = loadShader((char*)"res/vertex.shader").c_str();


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



std::string loadShader(char* path)
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
    std::cout<<"Error: Unable to open shader\n";
    }

    return allLines;
}