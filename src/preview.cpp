#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include "main.h"
#include "preview.h"
#include "scene.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#define IMGUI_IMPL_OPENGL_LOADER_GLEW

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;
bool valChanged = false;

std::string currentTimeString() {
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() {
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda() {
    if (pbo) {
        deletePBO(&pbo);
    }
    if (displayImage) {
        deleteTexture(&displayImage);
    }
}

void initCuda() {
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO() {
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);

}

void errorCallback(int error, const char* description) {
    fprintf(stderr, "%s\n", description);
}

bool init() {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui


    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;

    //// Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}

static ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None | ImGuiWindowFlags_NoMove;
static bool ui_hide = false;

// LOOK: Un-Comment to check ImGui Usage
void RenderImGui(int windowWidth, int windowHeight)
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    // Dear imgui new frame
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;
    // Dear imgui define
    ImVec2 minSize(300.f, 220.f);
    ImVec2 maxSize((float)windowWidth * 0.5, (float)windowHeight * 0.5);
    ImGui::SetNextWindowSizeConstraints(minSize, maxSize);

    ImGui::SetNextWindowPos(ui_hide ? ImVec2(-1000.f, -1000.f) : ImVec2(0.0f, 0.0f));

    ImGui::Begin("Control Panel", 0, windowFlags);
    ImGui::SetWindowFontScale(1);

    ImGui::Text("press H to hide GUI completely.");
    if (ImGui::IsKeyPressed('H')) {
        ui_hide = !ui_hide;
    }

    ImGui::DragInt("Iterations", &ui_iterations, 1, 1, startupIterations);

    ImGui::Checkbox("Denoise", &ui_denoise);
    ImGui::SameLine();
    bool atros_gaussChanged = ImGui::Checkbox("Gaussian Approximation", &ui_atros_gauss);
    bool gaussianChanged = ImGui::Checkbox("Gaussian", &ui_gaussian);
    ImGui::SameLine();
    bool sharedChanged = ImGui::Checkbox("Enable Shared Memory", &ui_shared);

    bool filterSizeChanged = ImGui::DragInt("Filter Size", &ui_filterSize, 1, 0, 100);
    ImGui::DragFloat("Color Weight", &ui_colorWeight, 0.1f, 0.0f, 100.0f, "%.4f");
    ImGui::DragFloat("Normal Weight", &ui_normalWeight, 0.1f, 0.0f, 1.0f, "%.4f");
    ImGui::DragFloat("Position Weight", &ui_positionWeight, 0.1f, 0.0f, 1.0f, "%.4f");

    ImGui::Separator();

    ImGui::Checkbox("Show GBuffer", &ui_showGbuffer);

    ImGui::Separator();
    ImGui::Checkbox("Sort By Material", &imguiData->SortByMaterial);
    ImGui::Checkbox("Enable BVH", &imguiData->UseBVH);
    ImGui::Checkbox("Enable ACES Film", &imguiData->ACESFilm);
    ImGui::Checkbox("Enable Reinhard", &imguiData->Reinhard);
    ImGui::Checkbox("Disable Gamma Correction", &imguiData->NoGammaCorrection);
    ImGui::Checkbox("Sync", &imguiData->Sync);
    bool cacheFirstBounce = ImGui::Checkbox("Cache First Bounce", &imguiData->CacheFirstBounce);
    float availWidth = ImGui::GetContentRegionAvail().x;
    ImGui::SetNextItemWidth(availWidth * 0.25f);
    bool focalLengthChanged = ImGui::DragFloat("Focal Length", &imguiData->focalLength, 0.1f, 0.0f, 8.0f, "%.4f", ImGuiInputTextFlags_CallbackEdit);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(availWidth * 0.25f);
    bool apertureSizeChanged = ImGui::DragFloat("Aperture Size", &imguiData->apertureSize, 0.001f, 0.000f, 0.05f, "%.4f", ImGuiInputTextFlags_CallbackEdit);
    ImGui::SetNextItemWidth(availWidth * 0.25f);
    bool cameraPhiChanged = ImGui::DragFloat("Camera Phi", &imguiData->phi, 0.1f, -PI, PI, "%.4f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(availWidth * 0.25f);
    bool cameraThetaChanged = ImGui::DragFloat("Camera Theta", &imguiData->theta, 0.1f, 0.001f, PI - 0.001f, "%.4f");
    bool cameraLookAtChanged = ImGui::DragFloat3("Camera Look At", &imguiData->cameraLookAt.x, 0.1f, -200.0f, 200.0f, "%.4f");
    bool zoomChanged = ImGui::DragFloat("Zoom", &imguiData->zoom, 0.01f, 0.01f, 100.0f, "%.4f");

    if (ImGui::Button("Save image and exit")) {
        ui_saveAndExit = true;
    }
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Denoise average %.3f ms/frame", imguiData->DenoiseTime / ui_denoise_cnt);
    ImGui::Text("Denoise %.3f ms/frame", imguiData->DenoiseRealTime);
    ImGui::Text("Number of triangles %d", scene->geoms.size());
    ImGui::End();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (focalLengthChanged || apertureSizeChanged || cameraPhiChanged || cameraThetaChanged || cameraLookAtChanged || zoomChanged || cacheFirstBounce) {
        valChanged = true;
    }
    if (filterSizeChanged || atros_gaussChanged || gaussianChanged || sharedChanged) {
        ui_denoise_cnt = 0;
        imguiData->DenoiseTime = 0;
    }
}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

bool ValueChanged()
{
    return valChanged;
}

void ResetValueChanged()
{
    valChanged = false;
}

void mainLoop() {
    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();

        runCuda();

        std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

        // Draw imgui
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        RenderImGui(display_w, display_h);

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}
