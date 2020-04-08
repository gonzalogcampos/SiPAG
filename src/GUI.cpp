#include <GUI.h>

#include <imGUI/imgui.h>
#include <Values.h>

bool show_demo_window = true;
static float f = 0.0f;
static int counter = 0;

void GUIupdate()
{
    ImGui::NewFrame();

    ImGui::Begin("Control Panel");                          // Create a window called "Hello, world!" and append into it.

    ImGui::Text("Camera");               // Display some text (you can use a format strings too)
    ImGui::SliderFloat("Rotation", &c_Rotation, 0.0f, 6.3f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::SliderFloat("Distance", &c_Distance, 0.0f, 10.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::SliderFloat("Height", &c_Height, 0.0f, 10.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
        counter++;
    ImGui::SameLine();
    ImGui::Text("counter = %d", counter);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();

    ImGui::Render();
}