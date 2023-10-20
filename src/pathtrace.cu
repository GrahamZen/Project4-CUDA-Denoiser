#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <thrust/count.h>

#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "material.h"

#define ERRORCHECK 1
#define i2col(y, x, width) ((y) * (width) + (x))
#define BLOCK_SIZE 4
#define MAX_BLOCK_SIZE 16

#ifdef DEBUG
template<typename T>
void checkCudaMem(T* d_ptr, int size) {
    T* h_ptr = new T[size];
    cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    delete[] h_ptr;
}
#endif // DEBUG

// http://jcgt.org/published/0003/02/01/paper.pdf
__device__ __inline__ glm::vec2 signNotZero(glm::vec2 v) {
    return glm::vec2((v.x >= 0.f) ? 1.f : -1.f, (v.y >= 0.f) ? +1.f : -1.f);
}

__device__ glm::vec2 float32x3_to_oct(const glm::vec3& v) {
    glm::vec2 p = glm::vec2(v.x, v.y) * (1.f / (abs(v.x) + abs(v.y) + abs(v.z)));
    return (v.z <= 0.f) ? ((1.f - abs(glm::vec2(p.y, p.x))) * signNotZero(p)) : p;
}
__device__ glm::vec3 oct_to_float32x3(glm::vec2 e) {
    glm::vec3 v = glm::vec3(glm::vec2(e.x, e.y), 1.f - abs(e.x) - abs(e.y));
    if (v.z < 0) {
        glm::vec2 tmp = (1.f - abs(glm::vec2(v.y, v.x))) * signNotZero(glm::vec2(v.x, v.y));
        v.x = tmp.x; v.y = tmp.y;
    }
    return normalize(v);
}

__constant__ constexpr float GaussianKernel[5] = { .0625f, .25f, .375f, .25f, .0625f };

__global__ void ATrousFilterKern(glm::ivec2 resolution, const glm::vec3* dev_image, const GBufferPixel* gBuffer, glm::vec3* outputCol,
    int stepWidth, float c_phi, float n_phi, float p_phi)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < resolution.x && y < resolution.y) {
        const int index = x + (y * resolution.x);
        if (stepWidth == 0) {
            outputCol[index] = dev_image[index];
            return;
        }
        glm::vec3 sum(0.f);
        glm::vec3 cval = dev_image[index];
#if OCT_NORMAL
        glm::vec3 nval = oct_to_float32x3(gBuffer[index].normal);
#else
        glm::vec3 nval = gBuffer[index].normal;
#endif
        glm::vec3 pval = gBuffer[index].pos;
        float cum_w = 0.f;
        int tmpIdx = 0;
        const float inv_c_phi = __fdividef(1.0f, c_phi * c_phi);
        const float inv_n_phi = __fdividef(1.0f, n_phi * n_phi);
        const float inv_p_phi = __fdividef(1.0f, p_phi * p_phi);
        const float inv_stepWidth = __fdividef(1.0f, stepWidth * stepWidth);
        for (int i = -2; i < 3; i++) {
            for (int j = -2; j < 3; j++) {
                tmpIdx = (x + i * stepWidth) + (y + j * stepWidth) * resolution.x;
                if (tmpIdx < 0 || tmpIdx >= resolution.x * resolution.y)continue;
                glm::vec3 ctmp = dev_image[tmpIdx];
#if OCT_NORMAL
                glm::vec3 ntmp = oct_to_float32x3(gBuffer[tmpIdx].normal);
#else
                glm::vec3 ntmp = gBuffer[tmpIdx].normal;
#endif
                glm::vec3 ptmp = gBuffer[tmpIdx].pos;

                glm::vec3 t = cval - ctmp;
                float dist2 = dot(t, t);
                float c_w = glm::min(__expf(-(dist2)*inv_c_phi), 1.f);

                t = nval - ntmp;
                dist2 = glm::max(dot(t, t) * inv_stepWidth, 0.f);
                float n_w = glm::min(__expf(-(dist2)*inv_n_phi), 1.f);

                t = pval - ptmp;
                dist2 = dot(t, t);
                float p_w = glm::min(__expf(-(dist2)*inv_p_phi), 1.f);

                float weight = c_w * n_w * p_w * GaussianKernel[j + 2] * GaussianKernel[i + 2];
                sum += ctmp * weight;
                cum_w += weight;
            }
        }
        outputCol[index] = sum * __fdividef(1.0f, cum_w);
    }
}

__global__ void ATrousFilterGaussKern(glm::ivec2 resolution, const glm::vec3* dev_image, const GBufferPixel* gBuffer, glm::vec3* outputCol, int stepWidth)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < resolution.x && y < resolution.y) {
        const int index = x + (y * resolution.x);
        if (stepWidth == 0) {
            outputCol[index] = dev_image[index];
            return;
        }
        glm::vec3 sum(0.f);
        float cum_w = 0.f;
        int tmpIdx = 0;
        for (int i = -2; i < 3; i++) {
            for (int j = -2; j < 3; j++) {
                tmpIdx = (x + i * stepWidth) + (y + j * stepWidth) * resolution.x;
                if (tmpIdx < 0 || tmpIdx >= resolution.x * resolution.y)continue;
                glm::vec3 ctmp = dev_image[tmpIdx];

                float weight = GaussianKernel[j + 2] * GaussianKernel[i + 2];
                sum += ctmp * weight;
                cum_w += weight;
            }
        }
        outputCol[index] = sum * __fdividef(1.0f, cum_w);
    }
}

__global__ void ATrousFilterKernSharedSmallStWd(glm::ivec2 resolution, const glm::vec3* dev_image, const GBufferPixel* gBuffer, glm::vec3* outputCol,
    int stepWidth, float c_phi, float n_phi, float p_phi)
{
    extern __shared__ char sharedMemory[];

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= resolution.x && y >= resolution.y)
        return;
    const int arr_sizeX = blockDim.x + 4 * stepWidth;
    const int arr_size = arr_sizeX * (blockDim.y + 4 * stepWidth);
    glm::vec3* sharedImage = reinterpret_cast<glm::vec3*>(sharedMemory);
    GBufferPixel* sharedGBuffer = reinterpret_cast<GBufferPixel*>(&sharedMemory[arr_size * sizeof(glm::vec3)]);

    const int index = i2col(y, x, resolution.x);
    if (stepWidth == 0) {
        outputCol[index] = dev_image[index];
        return;
    }
    for (int i = 0; i < glm::ceil(float(arr_size) / float(blockDim.x * blockDim.y)); i++)
    {
        int tid = i * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        if (tid < arr_size)
        {
            int x = blockIdx.x * blockDim.x + tid % arr_sizeX - 2 * stepWidth;
            int y = blockIdx.y * blockDim.y + tid / arr_sizeX - 2 * stepWidth;
            if (x < 0 || x >= resolution.x || y < 0 || y >= resolution.y)
            {
                sharedImage[tid] = glm::vec3(0.f);
                sharedGBuffer[tid] = GBufferPixel{};
            }
            else {
                sharedImage[tid] = dev_image[i2col(y, x, resolution.x)];
                sharedGBuffer[tid] = gBuffer[i2col(y, x, resolution.x)];
            }
        }
    }
    __syncthreads();
    int tx = threadIdx.x + 2 * stepWidth;
    int ty = threadIdx.y + 2 * stepWidth;
    int tmpTIdx = i2col(ty, tx, arr_sizeX);
    glm::vec3 sum(0.f);
    glm::vec3 cval = sharedImage[tmpTIdx];
#if OCT_NORMAL
    glm::vec3 nval = oct_to_float32x3(sharedGBuffer[tmpTIdx].normal);
#else
    glm::vec3 nval = sharedGBuffer[tmpTIdx].normal;
#endif
    glm::vec3 pval = sharedGBuffer[tmpTIdx].pos;
    float cum_w = 0.f;
    int tmpIdx = 0;
    const float inv_c_phi = __fdividef(1.0f, c_phi * c_phi);
    const float inv_n_phi = __fdividef(1.0f, n_phi * n_phi);
    const float inv_p_phi = __fdividef(1.0f, p_phi * p_phi);
    const float inv_stepWidth = __fdividef(1.0f, stepWidth * stepWidth);
    for (int i = -2; i < 3; i++) {
        for (int j = -2; j < 3; j++) {
            tmpTIdx = i2col(ty + j * stepWidth, tx + i * stepWidth, arr_sizeX);
            tmpIdx = i2col(y + j * stepWidth, x + i * stepWidth, resolution.x);
            if (tmpIdx < 0 || tmpIdx >= resolution.x * resolution.y)continue;
            glm::vec3 ctmp = sharedImage[tmpTIdx];
#if OCT_NORMAL
            glm::vec3 ntmp = oct_to_float32x3(sharedGBuffer[tmpTIdx].normal);
#else
            glm::vec3 ntmp = sharedGBuffer[tmpTIdx].normal;
#endif
            glm::vec3 ptmp = sharedGBuffer[tmpTIdx].pos;

            glm::vec3 t = cval - ctmp;
            float dist2 = dot(t, t);
            float c_w = glm::min(__expf(-(dist2)*inv_c_phi), 1.f);

            t = nval - ntmp;
            dist2 = glm::max(dot(t, t) * inv_stepWidth, 0.f);
            float n_w = glm::min(__expf(-(dist2)*inv_n_phi), 1.f);

            t = pval - ptmp;
            dist2 = dot(t, t);
            float p_w = glm::min(__expf(-(dist2)*inv_p_phi), 1.f);

            float weight = c_w * n_w * p_w * GaussianKernel[j + 2] * GaussianKernel[i + 2];
            sum += ctmp * weight;
            cum_w += weight;
        }
    }
    outputCol[index] = sum * __fdividef(1.0f, cum_w);
}

__global__ void ATrousFilterKernSharedLargeStWd(glm::ivec2 resolution, const glm::vec3* dev_image, const GBufferPixel* gBuffer, glm::vec3* outputCol,
    int stepWidth, float c_phi, float n_phi, float p_phi)
{
    extern __shared__ char sharedMemory[];

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= resolution.x && y >= resolution.y)
        return;
    const int arr_sizeX = blockDim.x * 5;
    const int arr_size = blockDim.x * blockDim.y * 25;
    glm::vec3* sharedImage = reinterpret_cast<glm::vec3*>(sharedMemory);
    GBufferPixel* sharedGBuffer = reinterpret_cast<GBufferPixel*>(&sharedMemory[arr_size * sizeof(glm::vec3)]);

    const int index = i2col(y, x, resolution.x);
    if (stepWidth == 0) {
        outputCol[index] = dev_image[index];
        return;
    }
    int tmpIdx, tid;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    for (int i = -2; i < 3; i++) {
        for (int j = -2; j < 3; j++) {
            tmpIdx = i2col(y + j * stepWidth, x + i * stepWidth, resolution.x);
            tid = i2col(ty + (j + 2) * blockDim.x, tx + (i + 2) * blockDim.x, arr_sizeX);
            if (tmpIdx < 0 || tmpIdx >= resolution.x * resolution.y)
            {
                sharedImage[tid] = glm::vec3(0.f);
                sharedGBuffer[tid] = GBufferPixel{};
            }
            else {
                sharedImage[tid] = dev_image[tmpIdx];
                sharedGBuffer[tid] = gBuffer[tmpIdx];
            }

        }
    }
    __syncthreads();
    const int idx = i2col(ty + 2 * blockDim.x, tx + 2 * blockDim.x, arr_sizeX);
    int tmpTIdx = 0;
    glm::vec3 sum(0.f);
    glm::vec3 cval = sharedImage[idx];
#if OCT_NORMAL
    glm::vec3 nval = oct_to_float32x3(sharedGBuffer[idx].normal);
#else
    glm::vec3 nval = sharedGBuffer[idx].normal;
#endif
    glm::vec3 pval = sharedGBuffer[idx].pos;
    float cum_w = 0.f;
    const float inv_c_phi = __fdividef(1.0f, c_phi * c_phi);
    const float inv_n_phi = __fdividef(1.0f, n_phi * n_phi);
    const float inv_p_phi = __fdividef(1.0f, p_phi * p_phi);
    const float inv_stepWidth = __fdividef(1.0f, stepWidth * stepWidth);
    for (int i = -2; i < 3; i++) {
        for (int j = -2; j < 3; j++) {
            tmpTIdx = idx + j * blockDim.x * arr_sizeX + i * blockDim.x;
            tmpIdx = index + j * stepWidth * resolution.x + i * stepWidth;
            if (tmpIdx < 0 || tmpIdx >= resolution.x * resolution.y)continue;
            glm::vec3 ctmp = sharedImage[tmpTIdx];
#if OCT_NORMAL
            glm::vec3 ntmp = oct_to_float32x3(sharedGBuffer[tmpTIdx].normal);
#else
            glm::vec3 ntmp = sharedGBuffer[tmpTIdx].normal;
#endif
            glm::vec3 ptmp = sharedGBuffer[tmpTIdx].pos;

            glm::vec3 t = cval - ctmp;
            float dist2 = dot(t, t);
            float c_w = glm::min(__expf(-(dist2)*inv_c_phi), 1.f);

            t = nval - ntmp;
            dist2 = glm::max(dot(t, t) * inv_stepWidth, 0.f);
            float n_w = glm::min(__expf(-(dist2)*inv_n_phi), 1.f);

            t = pval - ptmp;
            dist2 = dot(t, t);
            float p_w = glm::min(__expf(-(dist2)*inv_p_phi), 1.f);

            float weight = c_w * n_w * p_w * GaussianKernel[j + 2] * GaussianKernel[i + 2];
            sum += ctmp * weight;
            cum_w += weight;
        }
    }
    outputCol[index] = sum * __fdividef(1.0f, cum_w);
}

__device__ __inline__ float gaussianDistrib(int x, int y, float sigmaSqr) {
    return __expf(-(x * x + y * y) * __fdividef(1.0f, 2.0f * sigmaSqr)) * __fdividef(1.0f, 2.0f * PI * sigmaSqr);
}

__global__ void  gaussianFilterKern(glm::ivec2 resolution, const glm::vec3* dev_image, glm::vec3* outputCol, int stepWidth, float sigmaSqr)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < resolution.x && y < resolution.y) {
        const int index = x + (y * resolution.x);
        if (stepWidth == 0) {
            outputCol[index] = dev_image[index];
            return;
        }
        glm::vec3 sum(0.f);
        float weightSum = 0.f;
        float weight = 0.f;
        int tmpIdx = 0;
        for (int i = -stepWidth; i <= stepWidth; i++) {
            for (int j = -stepWidth; j <= stepWidth; j++) {
                tmpIdx = x + i + (y + j) * resolution.x;
                if (tmpIdx < 0 || tmpIdx >= resolution.x * resolution.y)continue;
                glm::vec3 color = dev_image[tmpIdx];
                weight = gaussianDistrib(i, j, sigmaSqr);
                weightSum += weight;
                sum += color * weight;
            }
        }
        outputCol[index] = sum / weightSum;
    }
}

__global__ void gaussianFilterKernShared(glm::ivec2 resolution, const glm::vec3* dev_image, glm::vec3* outputCol, int stepWidth, float sigmaSqr) {
    extern __shared__ glm::vec3 shared_image[];
    const int arr_sizeX = blockDim.x + 2 * stepWidth;
    const int arr_size = arr_sizeX * (blockDim.y + 2 * stepWidth);
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int t_x = threadIdx.x + stepWidth;
    const int t_y = threadIdx.y + stepWidth;

    if (x >= resolution.x || y >= resolution.y) return;

    for (int i = 0; i < glm::ceil(float(arr_size) / float(blockDim.x * blockDim.y)); i++)
    {
        int tid = i * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        if (tid < arr_size)
        {
            int x = blockIdx.x * blockDim.x + tid % arr_sizeX - stepWidth;
            int y = blockIdx.y * blockDim.y + tid / arr_sizeX - stepWidth;
            if (x < 0 || x >= resolution.x || y < 0 || y >= resolution.y)
                shared_image[tid] = glm::vec3(0.f);
            else
                shared_image[tid] = dev_image[i2col(y, x, resolution.x)];
        }
    }

    __syncthreads();

    glm::vec3 sum = glm::vec3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    int tmpIdx = 0;
    float weight = 0;
    for (int i = -stepWidth; i <= stepWidth; i++) {
        for (int j = -stepWidth; j <= stepWidth; j++) {
            tmpIdx = i2col(y + i, x + j, resolution.x);
            if (tmpIdx < 0 || tmpIdx >= resolution.x * resolution.y)continue;
            weight = gaussianDistrib(i, j, sigmaSqr);
            sum += weight * shared_image[i2col(t_y + i, t_x + j, arr_sizeX)];
            weightSum += weight;
        }
    }
    outputCol[i2col(y, x, resolution.x)] = sum / weightSum;
}

Denoiser::Denoiser(glm::ivec2 resolution) : resolution(resolution) {
    cudaMalloc(&dev_outputCol, resolution.x * resolution.y * sizeof(glm::vec3));
}

Denoiser::~Denoiser() {
    cudaFree(dev_outputCol);
}


void Denoiser::filter(const glm::vec3* image, const GBufferPixel* gBuffer, int level, float c_phi, float n_phi, float p_phi, bool useSharedMemory, bool gaussApprox) {
    int stepWidth = 1 << level;
    if (useSharedMemory) {
        const dim3 blockSize2d(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 blocksPerGrid2d((resolution.x + blockSize2d.x - 1) / blockSize2d.x, (resolution.y + blockSize2d.y - 1) / blockSize2d.y);
        if (BLOCK_SIZE < stepWidth) {
            const int arr_sizeX = BLOCK_SIZE * 5;
            ATrousFilterKernSharedLargeStWd << <blocksPerGrid2d, blockSize2d, arr_sizeX* arr_sizeX* (sizeof(glm::vec3) + sizeof(GBufferPixel)) >> > (resolution, image, gBuffer, dev_outputCol, stepWidth, c_phi, n_phi, p_phi);
            checkCUDAError("ATrousFilterKernSharedLargeStWd");
        }
        else {
            const int arr_sizeX = BLOCK_SIZE + 4 * stepWidth;
            ATrousFilterKernSharedSmallStWd << <blocksPerGrid2d, blockSize2d, arr_sizeX* arr_sizeX* (sizeof(glm::vec3) + sizeof(GBufferPixel)) >> > (resolution, image, gBuffer, dev_outputCol, stepWidth, c_phi, n_phi, p_phi);
            checkCUDAError("ATrousFilterKernSharedSmallStWd");
        }
    }
    else {
        const dim3 blockSize2d(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
        const dim3 blocksPerGrid2d((resolution.x + blockSize2d.x - 1) / blockSize2d.x, (resolution.y + blockSize2d.y - 1) / blockSize2d.y);
        if (gaussApprox)
            ATrousFilterGaussKern << <blocksPerGrid2d, blockSize2d >> > (resolution, image, gBuffer, dev_outputCol, stepWidth);
        else
            ATrousFilterKern << <blocksPerGrid2d, blockSize2d >> > (resolution, image, gBuffer, dev_outputCol, stepWidth, c_phi, n_phi, p_phi);
    }
}

void Denoiser::gaussianBlur(const glm::vec3* image, int stepWidth, bool useSharedMemory) {
    const dim3 blockSize2d(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 blocksPerGrid2d((resolution.x + blockSize2d.x - 1) / blockSize2d.x, (resolution.y + blockSize2d.y - 1) / blockSize2d.y);
    if (useSharedMemory) {
        const int arr_sizeX = BLOCK_SIZE + 2 * stepWidth;
        gaussianFilterKernShared << <blocksPerGrid2d, blockSize2d, arr_sizeX* arr_sizeX * sizeof(glm::vec3) >> > (resolution, image, dev_outputCol, stepWidth, stepWidth * stepWidth * 0.11f);
        checkCUDAError("gaussianFilterKernShared");
    }
    else {
        const dim3 blockSize2d(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
        const dim3 blocksPerGrid2d((resolution.x + blockSize2d.x - 1) / blockSize2d.x, (resolution.y + blockSize2d.y - 1) / blockSize2d.y);
        gaussianFilterKern << <blocksPerGrid2d, blockSize2d >> > (resolution, image, dev_outputCol, stepWidth, stepWidth * stepWidth * 0.11f);
        checkCUDAError("gaussianFilterKern");
    }
}


//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image, bool acesFilm, bool NoGammaCorrection, bool Reinhard) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] / (float)iter;

        if (acesFilm)
            pix = pix * (pix * (pix * 2.51f + 0.03f) + 0.024f) / (pix * (pix * 3.7f + 0.078f) + 0.14f);
        if (Reinhard)
            pix = pix / (1.f + pix);
        if (!NoGammaCorrection)
            pix = glm::pow(pix, glm::vec3(1.f / 2.2f));

        glm::ivec3 color = glm::ivec3(glm::clamp(pix, 0.f, 1.f) * 255.0f);
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
#if OCT_NORMAL
        glm::vec3 normalRGB = (oct_to_float32x3(gBuffer[index].normal) * 0.5f + 0.5f) * 256.f;
#else
        glm::vec3 normalRGB = (gBuffer[index].normal * 0.5f + 0.5f) * 256.f;
#endif
        pbo[index].w = 0;
        pbo[index].x = normalRGB.x;
        pbo[index].y = normalRGB.y;
        pbo[index].z = normalRGB.z;
    }
}
static Scene* hst_scene = nullptr;
static GuiDataContainer* guiData = nullptr;
glm::vec3* dev_image = nullptr;
glm::vec3* dev_image_denoised = nullptr;
static TriangleDetail* dev_geoms = nullptr;
static Material* dev_materials = nullptr;
static PathSegment* dev_paths = nullptr;
static PathSegment* dev_paths_terminated = nullptr;
static int* dev_materialIsectIndices = nullptr;
static int* dev_materialIsectIndicesCache = nullptr;
static int* dev_materialSegIndices = nullptr;
static TBVHNode* dev_tbvhNodes = nullptr;
static thrust::device_ptr<PathSegment> dev_paths_thrust;
static thrust::device_ptr<PathSegment> dev_paths_terminated_thrust;
static thrust::device_ptr<int> dev_materialIsectIndices_thrust;
static thrust::device_ptr<int> dev_materialSegIndices_thrust;
static ShadeableIntersection* dev_intersections = nullptr;
static ShadeableIntersection* dev_intersections_cache = nullptr;
GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData, Scene* scene)
{
    guiData = imGuiData;
    guiData->TracedDepth = scene->state.traceDepth;
    guiData->SortByMaterial = false;
    guiData->UseBVH = true;
    guiData->ACESFilm = false;
    guiData->NoGammaCorrection = false;
    guiData->CacheFirstBounce = scene->settings.trSettings.cacheFirstBounce;
    guiData->focalLength = scene->state.camera.focalLength;
    guiData->apertureSize = scene->state.camera.apertureSize;
    guiData->theta = 0.f;
    guiData->phi = 0.f;
    guiData->cameraLookAt = scene->state.camera.lookAt;
    guiData->zoom = 1.f;
}

void UpdateDataContainer(GuiDataContainer* imGuiData, Scene* scene, float zoom, float theta, float phi)
{
    imGuiData->TracedDepth = scene->state.traceDepth;
    imGuiData->SortByMaterial = false;
    imGuiData->NoGammaCorrection = false;
    imGuiData->theta = theta;
    imGuiData->phi = phi;
    imGuiData->cameraLookAt = scene->state.camera.lookAt;
    imGuiData->zoom = zoom;
}

void pathtraceInit(Scene* scene) {
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_image_denoised, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_denoised, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_paths_terminated, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(TriangleDetail));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(TriangleDetail), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections_cache, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_materialIsectIndices, pixelcount * sizeof(int));
    cudaMalloc(&dev_materialSegIndices, pixelcount * sizeof(int));

    cudaMalloc(&dev_materialIsectIndicesCache, pixelcount * sizeof(int));
    cudaMalloc(&dev_materialIsectIndicesCache, pixelcount * sizeof(int));

    dev_materialIsectIndices_thrust = thrust::device_ptr<int>(dev_materialIsectIndices);
    dev_materialSegIndices_thrust = thrust::device_ptr<int>(dev_materialSegIndices);

    dev_paths_thrust = thrust::device_ptr<PathSegment>(dev_paths);
    dev_paths_terminated_thrust = thrust::device_ptr<PathSegment>(dev_paths_terminated);

    cudaMalloc(&dev_tbvhNodes, 6 * hst_scene->tbvh.nodesNum * sizeof(TBVHNode));
    for (int i = 0; i < 6; i++)
    {
        cudaMemcpy(dev_tbvhNodes + i * hst_scene->tbvh.nodesNum, hst_scene->tbvh.nodes[i].data(), hst_scene->tbvh.nodesNum * sizeof(TBVHNode), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is nullptr
    cudaFree(dev_image_denoised);  // no-op if dev_image is nullptr
    cudaFree(dev_paths);
    cudaFree(dev_paths_terminated);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_materialIsectIndices);
    cudaFree(dev_materialIsectIndicesCache);
    cudaFree(dev_materialSegIndices);
    cudaFree(dev_intersections_cache);
    cudaFree(dev_tbvhNodes);
    cudaFree(dev_gBuffer);

    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, Scene::Settings::CameraSettings settings)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < cam.resolution.x && y < cam.resolution.y) {
        float rx = 0.f;
        float ry = 0.f;
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, pathSegments->remainingBounces);
        thrust::uniform_real_distribution<float> u(-0.5, 0.5);
        thrust::uniform_real_distribution<float> u01(0.f, 1.f);
        segment.ray.origin = cam.position;
        segment.color = glm::vec3(0.f);
        segment.throughput = glm::vec3(1.f);
        if (settings.antiAliasing) {
            rx = u(rng);
            ry = u(rng);
        }

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + rx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + ry - (float)cam.resolution.y * 0.5f));
        if (settings.dof) {
            glm::vec3 forward = glm::normalize(glm::cross(cam.up, cam.right));
            float t = cam.focalLength / AbsDot(segment.ray.direction, forward);
            glm::vec3 randPt = cam.apertureSize * squareToDiskConcentric(glm::vec2(u01(rng), u01(rng)));
            glm::vec3 tmpRayOrigin = cam.position + randPt.x * cam.right + randPt.y * cam.up;
            glm::vec3 focusPoint = segment.ray.origin + t * segment.ray.direction;
            segment.ray.direction = glm::normalize(focusPoint - tmpRayOrigin);
            segment.ray.origin = tmpRayOrigin;
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth
    , int num_paths
    , PathSegment* pathSegments
    , TriangleDetail* geoms
    , TBVHNode* nodes
    , int geoms_size
    , int nodesNum
    , ShadeableIntersection* intersections
    , bool sortByMaterial
    , int* materialIndices
    , bool useBVH
    , cudaTextureObject_t cubemap
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];
#if DISABLE_COMPACTION
        if (pathSegment.pixelIndex < 0 || pathSegment.remainingBounces <= 0)return;
#endif
        float3 tmp_t;
        float3 t;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;

        // naive parse through global geoms
        if (useBVH)
            tmp_t = sceneIntersectionTest(geoms, nodes, nodesNum, pathSegment.ray, hit_geom_index, cubemap);
        else {
            for (int i = 0; i < geoms_size; i++)
            {
                TriangleDetail& tri = geoms[i];
                t = triangleIntersectionTest(tri, pathSegment.ray);
                // TODO: add more intersection tests here... triangle? metaball? CSG?

                // Compute the minimum t from the intersection tests to determine what
                // scene geometry object was hit first.
                if (t.x > 0.0f && t_min > t.x)
                {
                    tmp_t = t;
                    t_min = t.x;
                    hit_geom_index = i;
                }
            }
        }


        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            materialIndices[path_index] = -1;
        }
        else
        {
            //The ray hits something
            t = tmp_t;
            t_min = t.x;
            float w = 1 - t.y - t.z;
            TriangleDetail& tri = geoms[hit_geom_index];
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = tri.materialid;
            intersections[path_index].uv = t.y * tri.uv0 + t.z * tri.uv1 + w * tri.uv2;
            intersections[path_index].pos = getPointOnRay(pathSegment.ray, t_min);
            intersections[path_index].woW = -pathSegment.ray.direction;
            intersections[path_index].surfaceNormal =
                glm::normalize(multiplyMV(tri.t.invTranspose, glm::vec4(t.y * tri.normal0 + t.z * tri.normal1 + w * tri.normal2, 0.f)));
            glm::vec4 tangent = t.y * tri.tangent0 + t.z * tri.tangent1 + w * tri.tangent2;
            float tmpTanw = tangent[3];
            tangent[3] = 0.f;
            intersections[path_index].tangent = glm::vec4(glm::normalize(multiplyMV(tri.t.invTranspose, tangent)), tmpTanw);
            if (sortByMaterial)
                materialIndices[path_index] = intersections[path_index].materialId;
        }
    }
}

__global__ void shadeMaterial(
    int iter
    , int depth
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
    , cudaTextureObject_t envMap
    , Scene::Settings::TransferableSettings settings
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment& pSeg = pathSegments[idx];
#if DISABLE_COMPACTION
        if (pSeg.pixelIndex < 0 || pSeg.remainingBounces <= 0)return;
#endif
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pSeg.remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];

            // If the material indicates that the object was a light, "light" the ray
            glm::vec3 emissiveColor = getEmissiveFactor(material, intersection.uv);
            if (material.type == Material::Type::LIGHT) {
                pSeg.color = pSeg.throughput * (glm::vec3(material.pbrMetallicRoughness.baseColorFactor) * material.emissiveFactor * material.emissiveStrength);
                pSeg.remainingBounces = 0;
            }
            else if (glm::length(emissiveColor) > EPSILON) {
                pSeg.color = pSeg.throughput * emissiveColor * material.emissiveFactor * material.emissiveStrength;
                pSeg.remainingBounces = 0;
            }
            else {
                if (settings.testNormal) {
                    glm::vec3 nor = intersection.surfaceNormal;
                    if (material.normalTexture.index != -1) {
                        nor = sampleTexture(material.normalTexture.cudaTexObj, intersection.uv);
                        nor = glm::normalize(nor) * 2.f - 1.f;
                        nor = glm::normalize((glm::mat3(glm::vec3(intersection.tangent), glm::cross(intersection.surfaceNormal, glm::vec3(intersection.tangent)) * intersection.tangent[3], intersection.surfaceNormal)) * nor);
                    }
                    pSeg.color = nor * 0.5f + 0.5f;
                    pSeg.remainingBounces = 0;
                }
                else if (settings.testIntersect) {
                    pSeg.color = settings.testColor;
                    pSeg.remainingBounces = 0;
                }
                else {
                    float ao{ 1.f };
                    if (material.occlusionTexture.index != -1) {
                        auto aoData = tex2D<float4>(material.occlusionTexture.cudaTexObj, intersection.uv.x, intersection.uv.y);
                        ao = aoData.x;
                    }
                    glm::vec3 nor = intersection.surfaceNormal;
                    if (material.normalTexture.index != -1) {
                        nor = sampleTexture(material.normalTexture.cudaTexObj, intersection.uv);
                        nor = glm::normalize(nor) * 2.f - 1.f;
                        nor = glm::normalize((glm::mat3(glm::vec3(intersection.tangent), glm::cross(intersection.surfaceNormal, glm::vec3(intersection.tangent)) * intersection.tangent[3], intersection.surfaceNormal)) * nor);
                    }
                    BsdfSample sample;
                    auto bsdf = ao * sample_f(material, settings.isProcedural, settings.scale, nor, intersection.uv, intersection.woW, glm::vec3(u01(rng), u01(rng), u01(rng)), sample);
                    if (sample.pdf <= 0) {
                        pSeg.remainingBounces = 0;
                        pSeg.pixelIndex = -1;
                    }
                    else {
                        pSeg.remainingBounces -= 1;
                        pSeg.throughput *= bsdf / sample.pdf * AbsDot(intersection.surfaceNormal, sample.wiW);
                        pSeg.ray = SpawnRay(intersection.pos, sample.wiW);
                    }
                }
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else if (settings.envMapEnabled) {
            pSeg.color += pSeg.throughput * sampleEnvTexture(envMap, sampleSphericalMap(pSeg.ray.direction));
            pSeg.remainingBounces = 0;
        }
        else {
            pSeg.color = glm::vec3(0.0f);
            pSeg.remainingBounces = 0;
            pSeg.pixelIndex = -1;
        }
    }
}

__global__ void generateGBuffer(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    GBufferPixel* gBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        //gBuffer[idx].t = shadeableIntersections[idx].t;
#if OCT_NORMAL
        gBuffer[idx].normal = float32x3_to_oct(shadeableIntersections[idx].surfaceNormal);
#else
        gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
#endif
        gBuffer[idx].pos = shadeableIntersections[idx].pos;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Pathtracing Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * NEW: For the first depth, generate geometry buffers (gbuffers)
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally:
    //     * if not denoising, add this iteration's results to the image
    //     * TODO: if denoising, run kernels that take both the raw pathtraced result and the gbuffer, and put the result in the "pbo" from opengl

    // TODO: perform one iteration of path tracing

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, hst_scene->settings.camSettings);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    auto dev_paths_terminated_end = dev_paths_terminated_thrust;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    if (!hst_scene->state.isCached && guiData->CacheFirstBounce) {
        hst_scene->state.isCached = true;
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , dev_tbvhNodes
            , hst_scene->geoms.size()
            , hst_scene->tbvh.nodesNum
            , dev_intersections_cache
            , guiData->SortByMaterial
            , dev_materialIsectIndices
            , guiData->UseBVH
            , hst_scene->cubemap.texObj
            );
        checkCUDAError("trace one bounce");
        if (guiData->SortByMaterial) {
            cudaMemcpy(dev_materialSegIndices, dev_materialIsectIndices, num_paths * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dev_materialIsectIndicesCache, dev_materialIsectIndices, num_paths * sizeof(int), cudaMemcpyDeviceToDevice);
            thrust::sort_by_key(dev_materialIsectIndices_thrust, dev_materialIsectIndices_thrust + num_paths, dev_intersections_cache);
        }
    }
    // Empty gbuffer
    cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));


    bool iterationComplete = false;
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
    while (!iterationComplete) {

        // tracing
        if (depth == 0 && hst_scene->state.isCached) {
            cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            if (guiData->SortByMaterial) {
                cudaMemcpy(dev_materialSegIndices, dev_materialIsectIndicesCache, pixelcount * sizeof(int), cudaMemcpyDeviceToDevice);
                thrust::sort_by_key(dev_materialSegIndices_thrust, dev_materialSegIndices_thrust + pixelcount, dev_paths_thrust);
            }
        }
        else {
            // clean shading chunks
            cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , dev_tbvhNodes
                , hst_scene->geoms.size()
                , hst_scene->tbvh.nodesNum
                , dev_intersections
                , guiData->SortByMaterial
                , dev_materialIsectIndices
                , guiData->UseBVH
                , hst_scene->cubemap.texObj
                );
            checkCUDAError("trace one bounce");
            if (guiData->SortByMaterial) {
                cudaMemcpy(dev_materialSegIndices, dev_materialIsectIndices, num_paths * sizeof(int), cudaMemcpyDeviceToDevice);
                thrust::sort_by_key(dev_materialIsectIndices_thrust, dev_materialIsectIndices_thrust + num_paths, dev_intersections);
                thrust::sort_by_key(dev_materialSegIndices_thrust, dev_materialSegIndices_thrust + num_paths, dev_paths_thrust);
            }
        }
        if (depth == 0) {
            generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
        }

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            hst_scene->envMapTexture.cudaTexObj,
            hst_scene->settings.trSettings);
        depth++;
        // gather valid terminated paths
#if DISABLE_COMPACTION

        int num_paths_valid = thrust::count_if(thrust::device, dev_paths_thrust, dev_paths_thrust + num_paths,
            [] __host__  __device__(const PathSegment & p) { return p.pixelIndex >= 0 && p.remainingBounces > 0; });
        iterationComplete = (num_paths_valid == 0); // TODO: should be based off stream compaction results.
#else
        dev_paths_terminated_end = thrust::remove_copy_if(dev_paths_thrust, dev_paths_thrust + num_paths, dev_paths_terminated_end,
            [] __host__  __device__(const PathSegment & p) { return !(p.pixelIndex >= 0 && p.remainingBounces == 0); });
        int num_paths_valid = dev_paths_terminated_end - dev_paths_terminated_thrust;
        auto end = thrust::remove_if(dev_paths_thrust, dev_paths_thrust + num_paths,
            [] __host__  __device__(const PathSegment & p) { return p.pixelIndex < 0 || p.remainingBounces <= 0; });
        num_paths = end - dev_paths_thrust;

        iterationComplete = (num_paths == 0); // TODO: should be based off stream compaction results.
#endif
        if (guiData != nullptr)
        {
            guiData->TracedDepth = depth;
        }
    }
#if DISABLE_COMPACTION
    dev_paths_terminated_end = thrust::remove_copy_if(dev_paths_thrust, dev_paths_thrust + num_paths, dev_paths_terminated_end,
        [] __host__  __device__(const PathSegment & p) { return !(p.pixelIndex >= 0 && p.remainingBounces == 0); });
#endif

    // Assemble this iteration and apply it to the image
    int num_paths_valid = dev_paths_terminated_end - dev_paths_terminated_thrust;
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths_valid, dev_image, dev_paths_terminated);

    ///////////////////////////////////////////////////////////////////////////

    // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
    // Otherwise, screenshots are also acceptable.

    checkCUDAError("pathtrace");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4* pbo, int iter) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image, guiData->ACESFilm, guiData->NoGammaCorrection, guiData->Reinhard);
}

void showDeNoisedImage(uchar4* pbo, int iter) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image_denoised, guiData->ACESFilm, guiData->NoGammaCorrection, guiData->Reinhard);
}
