#include <OClock.h>
#include <Values.h>
#include <Console.h>

void OClock::start()
{
    lastTime = std::chrono::high_resolution_clock::now();
};

float OClock::getElapsedTime()
{
    return elapsedTime.count();
};

float OClock::step()
{
    std::chrono::time_point<std::chrono::high_resolution_clock> currentTime = std::chrono::high_resolution_clock::now();
    elapsedTime = currentTime - lastTime;
    lastTime = currentTime;
    float fps = 1/elapsedTime.count();
    cPrint("FPS: " + cString(fps) + "\n", 2);
    return elapsedTime.count();
}

bool OClock::canContinue()
{
    /*
    std::chrono::time_point<std::chrono::high_resolution_clock> currentTime = std::chrono::high_resolution_clock::now();

    elapsedTime = currentTime - lastTime;

    double minTime = 1.0/values::sys_FPS;
    if(elapsedTime.count() > minTime)
    {
        lastTime = currentTime;
        return true;
    }
    */
    return false;
}