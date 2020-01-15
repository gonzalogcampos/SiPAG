#include <Render.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <iostream>

Render::Render()
{

}

Render::~Render()
{

}


void Render::start()
{ 
	int argc = 1;
  	char *argv[1] = {(char*)""};
	glutInit(&argc, argv);
	glutInitWindowSize(720, 720);
  	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  	window = glutCreateWindow("SiPAG");

	GLenum res = glewInit();
    if (res != GLEW_OK)
    {
        std::cout<<"Error on inti OpenGL\n";
    }
    glEnable( GL_DEBUG_OUTPUT );
    //glDebugMessageCallback( (GLDEBUGPROC) MessageCallback, 0 );

	    //Enable Z-Buffer 
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS); 
	glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

	    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Render::draw()
{

}

void Render::close()
{
	glutDestroyWindow(window);
}

