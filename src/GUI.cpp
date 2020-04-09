#include <GUI.h>

#include <imGUI/imgui.h>
#include <imGUI/imgui_impl_glfw.h>
#include <imGUI/imgui_impl_opengl3.h>
#include <Values.h>
#include <CudaControler.h>
#include <Render.h>
#include <Console.h>


static int MaxParticles = e_MaxParticles;
static bool esferico = true;
static bool lineal = false;
static bool espiral = false;
static bool defShader = true;



void GUIupdate()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();

    ImGui::NewFrame();

    ImGui::Begin("Control Panel");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    
    gui_System();
    gui_Emitter();
    gui_Render();
    gui_Particle();
    gui_Wind();

    ImGui::End();
    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

void gui_System()
{
    ImGui::Separator();
    ImGui::BeginGroup();
    ImGui::Text("System");
    ImGui::InputInt("Cuda Block Size", &cu_BlockSize);
    ImGui::EndGroup();

}


void gui_Emitter()
{
    ImGui::Separator();
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
    ImGui::Separator();
    ImGui::Text("Particle");
    ImGui::SliderFloat("Life",                  &p_LifeTime,            0.f,    5.f);
    ImGui::SliderFloat("Random Life",           &p_RLifeTime,           0.f,    5.f);
    ImGui::SliderFloat("Size min",              &p_minSize,             .01f,   5.f);
    ImGui::SliderFloat("Size inc",              &p_incSize,             0.f,    5.f);   //%per second size improves
    ImGui::SliderFloat3("Velocity",             p_InitVelocity,         -5.f,   5.f);    //Init velocity
    /*ImGui::SameLine();
    if(ImGui::Button("Set 0"))
    {
        p_InitVelocity[0] =  0.f;
        p_InitVelocity[1] =  0.f;
        p_InitVelocity[2] =  0.f;
    }*/
    ImGui::SliderFloat3("Rand velocity",        p_RInitVelocity,        -5.f,   5.f);    //Random init velocity
    /*ImGui::SameLine();
    if(ImGui::Button("Set 0"))
    {
        p_RInitVelocity[0] =  0.f;
        p_RInitVelocity[1] =  0.f;
        p_RInitVelocity[2] =  0.f;
    }*/
    ImGui::SliderFloat("Velocity decay",        &p_VelocityDecay,       0.f,    5.f);
}

void gui_Render()
{
    ImGui::Separator();
    ImGui::Text("Render");
    ImGui::SliderFloat("Camera Rotation",       &c_Rotation,            0.0f,   6.3f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::SliderFloat("Camera Distance",       &c_Distance,            0.0f,   100.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::SliderFloat("Camera Height",         &c_Height,              -20.0f, 20.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::SameLine();
    if(ImGui::Button("Set 0"))
        c_Height =  0.f;
    
    ImGui::ColorEdit3("Background Color",       r_BackgroundColor);

    if(ImGui::Button("Change Shader"))
    {   
        defShader = !defShader;
        Render::getInstance()->changeShader();
    }
    if(defShader)
    {
        ImGui::Checkbox("Use R channel as alpha", &r_RasAlpha);
        if(r_RasAlpha)
            	ImGui::ColorEdit3("Particle Color", r_DefaultColor);
        ImGui::SliderFloat("Opacity", &r_MaxOpacity, 0.f, 1.f);
        ImGui::SliderFloat("T Opacity growing", &r_TimeOpacityGrowing, 0.f, 1.f);
        ImGui::SliderFloat("T Opacity decreasing", &r_TimeOpacityDecreasing, 0.f, 1.f);
    }else{   
        ImGui::ColorEdit4("Dots Color", r_DotsColor);
        ImGui::ColorEdit4("Wire Color", r_WiresColor);
    }
}

void gui_Wind()
{
    ImGui::Separator();
    ImGui::Text("Wind");
    ImGui::SliderFloat3("Constant Wind",w_Constant, -5.f, 5.f);
    /*ImGui::SameLine();
    if(ImGui::Button("Set 0"))
    {
        w_Constant[0] =  0.f;
        w_Constant[1] =  0.f;
        w_Constant[2] =  0.f;
    }*/

    ImGui::Checkbox("Wind noise 1", &w_1);
    if(w_1)
    {
        ImGui::SliderFloat3("Amplitude", w_1Amp, 0.f, 5.f);
        ImGui::SliderFloat("Size", &w_1Size, 0.f, 10.f);
        ImGui::SliderFloat("Lacunarity", &w_1lacunarity, 0.f, 10.f);
        ImGui::SliderFloat("Decay", &w_1decay, 0.f, 10.f);
        ImGui::SliderInt("N iterations", &w_1n, 0.f, 10.f);
    }

    ImGui::Checkbox("Wind noise 2", &w_2);
    if(w_2)
    {
        ImGui::SliderFloat3("Amplitude", w_2Amp, 0.f, 5.f);
        ImGui::SliderFloat("Size", &w_2Size, 0.f, 100.f);
        ImGui::SliderFloat("Lacunarity", &w_2lacunarity, 0.f, 10.f);
        ImGui::SliderFloat("Decay", &w_2decay, 0.f, 10.f);
        ImGui::SliderInt("N iterations", &w_2n, 0.f, 10.f);
    }

    
}


void changeSize()
{
    e_MaxParticles = MaxParticles;
    CudaControler::getInstance()->resize();
    Render::getInstance()->resize();
}
