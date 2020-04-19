#include <GUI.h>

#include <imGUI/imgui.h>
#include <imGUI/imgui_impl_glfw.h>
#include <imGUI/imgui_impl_opengl3.h>
#include <Values.h>
#include <CudaControler.h>
#include <CPUControler.h>
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
    ImGui::SetWindowPos(ImVec2(20,20));
    ImGui::SetWindowSize(ImVec2(520, 680));

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
    std::string but = "Pause";
    if(paused)
        but = "Play";

    ImGui::Separator();
    ImGui::BeginGroup();
    ImGui::Text("System");

    if(ImGui::Button(but.c_str()))
        paused = !paused;

    ImGui::SameLine();
    ImGui::Checkbox("Use CUDA", &GPU_Computing);

    if(GPU_Computing)
    {
        ImGui::InputInt("Cuda Block Size", &cu_BlockSize);
        ImGui::Checkbox("Pasate Constants", &cu_CopyConstants);
        ImGui::Checkbox("Update Random Kernel", &cu_UpdateRandomKernel);
    }

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
    ImGui::SliderInt("Emision Frec", &e_EmissionFrec, 0, 1000);
    ImGui::Checkbox("Spherical", &esferico);
    if(esferico)
    {
        e_Type = 0;
        lineal = false;
        espiral = false;
        ImGui::SliderFloat("Emitter Radious", &e_Length, 0.f, 10.f);
    }
    ImGui::Checkbox("Lineal", &lineal);
    if(lineal)
    {
        e_Type = 1;
        esferico = false;
        espiral = false;
        ImGui::SliderFloat("Emitter length", &e_Length, 0.f, 10.f);
    }
    ImGui::Checkbox("Spiral", &espiral);
    if(espiral)
    {
        e_Type = 2;
        lineal = false;
        esferico = false;
        ImGui::SliderFloat("Emitter length", &e_Length, 0.f, 10.f);
    }
    
}

void gui_Particle()
{
    ImGui::Separator();
    ImGui::Text("Particle");
    ImGui::SliderFloat("Life",                  &p_LifeTime,            0.f,    5.f);
    ImGui::SliderFloat("Random Life",           &p_RLifeTime,           0.f,    5.f);
    ImGui::SliderFloat3("Velocity",             p_InitVelocity,         -5.f,   5.f);    //Init velocity
    ImGui::SliderFloat3("Rand velocity",        p_RInitVelocity,        -1.f,   1.f);    //Random init velocity
    ImGui::SliderFloat("Velocity decay",        &p_VelocityDecay,       0.f,    5.f);
}

void gui_Render()
{
    ImGui::Separator();
    ImGui::Text("Render");
    ImGui::Checkbox("Enable Render", &r_enable);
    ImGui::ColorEdit3("Background Color", r_BackgroundColor);
    if(r_enable)
    {
        ImGui::Checkbox("Auto Rotation", &c_autoRotation);
        if(c_autoRotation)
            ImGui::SliderFloat("Camera Rotation Velocity", &c_autoRotationV, 0.f, 5.f);
        else  
            ImGui::SliderFloat("Camera Rotation", &c_Rotation, -6.3f, 6.3f);            // Edit 1 float using a slider from 0.0f to 1.0f
        
        ImGui::SliderFloat("Camera Distance",       &c_Distance, 0.0f, 100.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::SliderFloat("Camera Height",         &c_Height, -20.0f, 20.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::SameLine();
        if(ImGui::Button("Set 0"))
            c_Height =  0.f;
        

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
            ImGui::SliderFloat("Size min",              &p_minSize,             .01f,   5.f);
            ImGui::SliderFloat("Size inc",              &p_incSize,             0.f,    5.f);   //%per second size improves
            ImGui::SliderFloat("Opacity", &r_MaxOpacity, 0.f, 1.f);
            ImGui::SliderFloat("T Opacity growing", &r_TimeOpacityGrowing, 0.f, 1.f);
            ImGui::SliderFloat("T Opacity decreasing", &r_TimeOpacityDecreasing, 0.f, 1.f);
        }else{   
            ImGui::ColorEdit4("Dots Color", r_DotsColor);
        }
    }
}

void gui_Wind()
{
    ImGui::Separator();
    ImGui::Text("Wind");
    ImGui::SliderFloat3("Constant Wind",w_Constant, -1.f, 1.f);

    if(GPU_Computing)
    {
        ImGui::SliderFloat("Wind motion", &timeEv, 0.f, 10.f);
        ImGui::Checkbox("Wind noise 1", &w_1);
        if(w_1)
        {
            ImGui::SliderFloat3("1 Amplitude", w_1Amp, 0.f, 1.f);
            ImGui::SliderFloat("1 Size", &w_1Size, 0.f, 2.f);
            ImGui::SliderFloat("1 Lacunarity", &w_1lacunarity, 0.f, 3.f);
            ImGui::SliderFloat("1 Decay", &w_1decay, 0.f, 1.f);
            ImGui::SliderInt("1 Octaves", &w_1n, 0.f, 10.f);
        }

        ImGui::Checkbox("Wind noise 2", &w_2);
        if(w_2)
        {
            ImGui::SliderFloat3("2 Amplitude", w_2Amp, 0.f, 1.f);
            ImGui::SliderFloat("2 Size", &w_2Size, 0.f, 2.f);
            ImGui::SliderFloat("2 Lacunarity", &w_2lacunarity, 0.f, 3.f);
            ImGui::SliderFloat("2 Decay", &w_2decay, 0.f, 1.f);
            ImGui::SliderInt("2 Octaves", &w_2n, 0.f, 10.f);
        }
    }else
    {
        ImGui::Text("Wind noise is not available in CPU");
    }
    

    
}


void changeSize()
{
    e_MaxParticles = MaxParticles;
    CudaControler::getInstance()->resize();
    CPUControler::getInstance()->resize();
    Render::getInstance()->resize();
}
