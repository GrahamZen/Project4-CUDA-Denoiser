#pragma once

#include <vector>

class Scene;
class GuiDataContainer;

void InitDataContainer(GuiDataContainer* guiData, Scene* scene);
void UpdateDataContainer(GuiDataContainer* imGuiData, Scene* scene, float zoom, float theta, float phi);
void pathtraceInit(Scene* scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void showDeNoisedImage(uchar4* pbo, int iter);

struct GBufferPixel;

class Denoiser {
public:
    Denoiser(glm::ivec2 resolution);
    ~Denoiser();
    void filter(const glm::vec3* image, const GBufferPixel* gBuffer, int level, float c_phi, float n_phi, float p_phi, bool useSharedMemory, bool gaussApprox);
    void gaussianBlur(const glm::vec3* image, int stepWidth, bool useSharedMemory);
    glm::vec3* dev_outputCol;
private:
    glm::ivec2 resolution;
};