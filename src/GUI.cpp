#include <GUI.h>

#include <imGUI/imgui.h>
#include <Values.h>
#include <CudaControler.h>
#include <Render.h>
#include <Console.h>


static int MaxParticles = e_MaxParticles;
static bool esferico = true;
static bool lineal = false;
static bool espiral = false;
void GUIupdate()
{
    ImGui::NewFrame();

    ImGui::Begin("Control Panel");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Separator();
    gui_System();
    ImGui::Separator();
    gui_Emitter();
    ImGui::Separator();
    gui_Render();
    ImGui::Separator();
    gui_Particle();
    ImGui::Separator();
    gui_Wind();

    ImGui::End();
    ImGui::Render();
}

void gui_System()
{
    ImGui::Text("System");
    ImGui::InputInt("Cuda Block Size", &cu_BlockSize);
}


void gui_Emitter()
{
    ImGui::Text("Emitter");
    ImGui::InputInt("Max particles", &MaxParticles);
    ImGui::SameLine();
    if(ImGui::Button("Change"))
        changeSize();
    ImGui::InputInt("Emision Frec", &e_EmissionFrec);
    ImGui::Checkbox("Spherical", &esferico);
    if(esferico)
    {
        e_Type = 0;
        lineal = false;
        espiral = false;
        ImGui::SliderFloat("Emitter Radious", &e_Length, 0.f, 5.f);
    }
    ImGui::Checkbox("Lineal", &lineal);
    if(lineal)
    {
        e_Type = 1;
        esferico = false;
        espiral = false;
        ImGui::SliderFloat("Emitter length", &e_Length, 0.f, 5.f);
    }
    ImGui::Checkbox("Spiral", &espiral);
    if(espiral)
    {
        e_Type = 2;
        lineal = false;
        esferico = false;
        ImGui::SliderFloat("Emitter length", &e_Length, 0.f, 5.f);
    }
    
}

void gui_Particle()
{
    ImGui::Text("Particle");
    ImGui::SliderFloat("Life",                  &p_LifeTime,            0.f,    5.f);
    ImGui::SliderFloat("Random Life",           &p_RLifeTime,           0.f,    5.f);
    ImGui::SliderFloat("Size min",              &p_minSize,             .01f,   5.f);
    ImGui::SliderFloat("Size inc",              &p_incSize,             0.f,    5.f);   //%per second size improves
    ImGui::SliderFloat("Opacity",               &p_Opacity,             0.f,    5.f);   //Opacity of the particle
    ImGui::SliderFloat("Opacity evolution",     &p_OpacityEvolution,    0.f,    5.f);   //% per second opacity decays
    ImGui::SliderFloat3("Velocity",             p_InitVelocity,         -5.f,   5.f);    //Init velocity
    ImGui::SliderFloat3("Rand velocity",        p_RInitVelocity,        -5.f,   5.f);    //Random init velocity
    ImGui::SliderFloat("Velocity decay",        &p_VelocityDecay,       0.f,    5.f);
}

void gui_Render()
{
    ImGui::Text("Render");
    if(ImGui::Button("Change Shader"))
        Render::getInstance()->changeShader();
    ImGui::SliderFloat("Camera Rotation", &c_Rotation, 0.0f, 6.3f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::SliderFloat("Camera Distance", &c_Distance, 0.0f, 100.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::SliderFloat("Camera Height", &c_Height, -20.0f, 20.0f);            // Edit 1 float using a slider from 0.0f to 1.0f


    ImGui::ColorEdit3("Background Color", r_BackgroundColor);
    ImGui::ColorEdit4("Dots Color", r_DotsColor);
    ImGui::ColorEdit4("Wire Color", r_WiresColor);
    ImGui::ColorEdit3("Particle Color", r_ParticleColor);
}

void gui_Wind()
{

}


void changeSize()
{
    e_MaxParticles = MaxParticles;
    CudaControler::getInstance()->resize();
    Render::getInstance()->resize();
}
