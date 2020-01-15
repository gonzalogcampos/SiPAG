#pragma once


class Render
{
    public:
        Render();
        ~Render();

        void start();
        void draw();
        void close();
    private:
        int window;
};