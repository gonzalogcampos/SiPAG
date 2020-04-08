#include <GUI.h>

#include <imGUI/imgui.h>
#include <Values.h>
#include <CudaControler.h>
#include <Render.h>
#include <Console.h>

static int MaxParticles = e_MaxParticles;
void GUIupdate()
{
    ImGui::NewFrame();

    ImGui::Begin("Control Panel");                          // Create a window called "Hello, world!" and append into it.
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    
    ImGui::Text("Emitter");
    ImGui::InputInt("Max particles", &MaxParticles);
    ImGui::SameLine();
    if(ImGui::Button("Change"))
        changeSize();

    ImGui::Text("Render");
    if(ImGui::Button("Change Shader"))
        Render::getInstance()->changeShader();
    ImGui::SliderFloat("Camera Rotation", &c_Rotation, 0.0f, 6.3f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::SliderFloat("Camera Distance", &c_Distance, 0.0f, 100.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::SliderFloat("Camera Height", &c_Height, 0.0f, 100.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    
    ImGui::Text("Particle");
    ImGui::SliderFloat("Life",                  &p_LifeTime,            0.f,    5.f);
    ImGui::SliderFloat("Random Life",           &p_RLifeTime,           0.f,    5.f);
    ImGui::SliderFloat("Size",                  &p_Size,                .01f,   5.f);
    ImGui::SliderFloat("Size evolution",        &p_SizeEvolution,       0.f,    5.f);   //%per second size improves
    ImGui::SliderFloat("Opacity",               &p_Opacity,             0.f,    5.f);   //Opacity of the particle
    ImGui::SliderFloat("Opacity evolution",     &p_OpacityEvolution,    0.f,    5.f);   //% per second opacity decays
    ImGui::SliderFloat("X velocity",            &p_InitVelocityX,       0.f,    5.f);   //X init velocity
    ImGui::SliderFloat("Y velocity",            &p_InitVelocityY,       0.f,    5.f);   //Y init velocity
    ImGui::SliderFloat("Z velocity",            &p_InitVelocityZ,       0.f,    5.f);   //Z init velocity
    ImGui::SliderFloat("X rand velocity",       &p_RInitVelocityX,      0.f,    5.f);   //X random in init velocity
    ImGui::SliderFloat("Y rand velocity",       &p_RInitVelocityY,      0.f,    5.f);   //Y random in init velocity
    ImGui::SliderFloat("Z rand velocity",       &p_RInitVelocityZ,      0.f,    5.f);   //Z random in init velocity
    ImGui::SliderFloat("Velocity decay",        &p_VelocityDecay,       0.f,    5.f);

    

    ImGui::End();

    ImGui::Render();
}

void changeSize()
{
    e_MaxParticles = MaxParticles;
    CudaControler::getInstance()->resize();
    Render::getInstance()->resize();
}