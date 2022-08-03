#ifndef SCRFD_TRT_H
#define SCRFD_TRT_H

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <map>
#include <chrono>
#include "face_detector.h"

#define BATCH_SIZE 1

using namespace nvinfer1;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

class SCRFD_TRT {
    public:
        ~SCRFD_TRT();
        SCRFD_TRT(std::string trt_path);
        int detect(cv::Mat image, std::vector<FaceObject>& faceobjects, float prob_threshold, float nms_threshold);
        int draw(cv::Mat& image, const std::vector<FaceObject>& faceobjects);
    
    private:
        void initAnchors();
        void initEngine(const std::string trt_path);
        float** anchors_stride_8;
        float** anchors_stride_16;
        float** anchors_stride_32;
        int num_anchors_stride_8;
        int num_anchors_stride_16;
        int num_anchors_stride_32;
        Logger gLogger;
        IExecutionContext* context;
        ICudaEngine* engine;
        IRuntime* runtime;

        void doInference(IExecutionContext& context, float* input, 
                            float* score_8, float* bbox_8, float* kps_8,
                            float* score_16, float* bbox_16, float* kps_16,
                            float* score_32, float* bbox_32, float* kps_32,
                            int batch_size);
        float** generate_anchors(int base_size, const float ratios[], const float scales[]);
        void generate_proposals(float** anchors, int num_anchors, int feat_stride, float* score_blob, float* bbox_blob, float* kps_blob, float prob_threshold, std::vector<FaceObject>& faceobjects, int output_score_size);

        // sizes
        const int INPUT_H = 640;
        const int INPUT_W = 640;
        const int INPUT_SIZE = 3 * INPUT_W * INPUT_H;

        const int OUTPUT_BBOX_8_SIZE = 8 * 80 * 80;
        const int OUTPUT_KPS_8_SIZE = 20 * 80 * 80;
        const int OUTPUT_SCORE_8_SIZE = 2 * 80 * 80;

        const int OUTPUT_BBOX_16_SIZE = 8 * 40 * 40;
        const int OUTPUT_KPS_16_SIZE = 20 * 40 * 40;
        const int OUTPUT_SCORE_16_SIZE = 2 * 40 * 40;

        const int OUTPUT_BBOX_32_SIZE = 8 * 20 * 20;
        const int OUTPUT_KPS_32_SIZE = 20 * 20 * 20;
        const int OUTPUT_SCORE_32_SIZE = 2 * 20 * 20;

        // names
        const char* INPUT_BLOB_NAME = "image";

        const char* OUTPUT_BBOX_8_BLOB_NAME = "bbox_8";
        const char* OUTPUT_KPS_8_BLOB_NAME = "kps_8";
        const char* OUTPUT_SCORE_8_BLOB_NAME = "score_8";

        const char* OUTPUT_BBOX_16_BLOB_NAME = "bbox_16";
        const char* OUTPUT_KPS_16_BLOB_NAME = "kps_16";
        const char* OUTPUT_SCORE_16_BLOB_NAME = "score_16";

        const char* OUTPUT_BBOX_32_BLOB_NAME = "bbox_32";
        const char* OUTPUT_KPS_32_BLOB_NAME = "kps_32";
        const char* OUTPUT_SCORE_32_BLOB_NAME = "score_32";
};

#endif