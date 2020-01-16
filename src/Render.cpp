#include <Render.h>

#include <GL/glew.h>
#include <iostream>



void Render::start()
{
	GLenum res = glewInit();
    if (res != GLEW_OK)
    {
        std::cout<<"Error on init OpenGL\n";
    }
    glEnable( GL_DEBUG_OUTPUT );
    //glDebugMessageCallback( (GLDEBUGPROC) MessageCallback, 0 );

	//Enable Z-Buffer 
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS); 
	glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
}

void Render::draw()
{
	// Clear the screen
    glClearColor(0.0, 0.0, 0.0, 1.0);
  	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Render::close()
{


}

