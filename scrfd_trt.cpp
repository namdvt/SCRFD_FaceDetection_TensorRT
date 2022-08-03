#include "scrfd_trt.h"

using namespace nvinfer1;

void qsort_descent_inplace(std::vector<FaceObject> &face_objects, int left, int right) {
    int i = left;
    int j = right;
    float p = face_objects[(left + right) / 2].prob;

    while (i <= j) {
        while (face_objects[i].prob > p) {
            i++;
        }

        while (face_objects[j].prob < p) {
            j--;
        }

        if (i <= j) {
            std::swap(face_objects[i], face_objects[j]);
            i++;
            j--;
        }
    }

    if (left < j) {
        qsort_descent_inplace(face_objects, left, j);
    }

    if (i < right) {
        qsort_descent_inplace(face_objects, i, right);
    }
}

void qsort_descent_inplace(std::vector<FaceObject> &face_objects) {
    if (face_objects.empty()) {
        return;
    }

    qsort_descent_inplace(face_objects, 0, face_objects.size() - 1);
}

float intersection_area(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void nms_sorted_bboxes(const std::vector<FaceObject> &face_objects, std::vector<int> &picked,
                              float nms_threshold) {
    picked.clear();

    const int n = face_objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = face_objects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const FaceObject &a = face_objects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const FaceObject &b = face_objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

int SCRFD_TRT::draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects)
{
    for (size_t i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];
        cv::rectangle(rgb, obj.rect, cv::Scalar(0, 255, 0));

        cv::circle(rgb, obj.landmark[0], 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(rgb, obj.landmark[1], 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(rgb, obj.landmark[2], 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(rgb, obj.landmark[3], 2, cv::Scalar(255, 255, 0), -1);
        cv::circle(rgb, obj.landmark[4], 2, cv::Scalar(255, 255, 0), -1);

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    cv::imwrite("output.png", rgb);

    return 0;
}

SCRFD_TRT::SCRFD_TRT(const std::string trt_path) {
    initEngine(trt_path);
    initAnchors();
}

SCRFD_TRT::~SCRFD_TRT() {
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
}

void SCRFD_TRT::initEngine(const std::string trt_path) {
    std::cout << "init scrfd_trt from "<< trt_path <<"\n";
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file(trt_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
}


void SCRFD_TRT::initAnchors() {
    // stride 8
    {
        int base_size = 16;
        float ratios[] = {1.0};
        float scales[] = {1.0, 2.0};
        anchors_stride_8 = generate_anchors(base_size, ratios, scales);
        num_anchors_stride_8 = sizeof(ratios) / sizeof(ratios[0]) * sizeof(scales) / sizeof(scales[0]);
    }
    // stride 16
    {
        int base_size = 64;
        float ratios[] = {1.0};
        float scales[] = {1.0, 2.0};
        anchors_stride_16 = generate_anchors(base_size, ratios, scales);
        num_anchors_stride_16 = sizeof(ratios) / sizeof(ratios[0]) * sizeof(scales) / sizeof(scales[0]);
    }
    // stride 32
    {
        int base_size = 256;
        float ratios[] = {1.0};
        float scales[] = {1.0, 2.0};
        anchors_stride_32 = generate_anchors(base_size, ratios, scales);
        num_anchors_stride_32 = sizeof(ratios) / sizeof(ratios[0]) * sizeof(scales) / sizeof(scales[0]);
    }
}


void SCRFD_TRT::doInference(IExecutionContext& context, float* input, 
                            float* score_8, float* bbox_8, float* kps_8,
                            float* score_16, float* bbox_16, float* kps_16,
                            float* score_32, float* bbox_32, float* kps_32,
                            int batch_size) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 10);
    void* buffers[10];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int input_index = engine.getBindingIndex(INPUT_BLOB_NAME);

    const int bbox_8_index = engine.getBindingIndex(OUTPUT_BBOX_8_BLOB_NAME);
    const int kps_8_index = engine.getBindingIndex(OUTPUT_KPS_8_BLOB_NAME);
    const int score_8_index = engine.getBindingIndex(OUTPUT_SCORE_8_BLOB_NAME);

    const int bbox_16_index = engine.getBindingIndex(OUTPUT_BBOX_16_BLOB_NAME);
    const int kps_16_index = engine.getBindingIndex(OUTPUT_KPS_16_BLOB_NAME);
    const int score_16_index = engine.getBindingIndex(OUTPUT_SCORE_16_BLOB_NAME);

    const int bbox_32_index = engine.getBindingIndex(OUTPUT_BBOX_32_BLOB_NAME);
    const int kps_32_index = engine.getBindingIndex(OUTPUT_KPS_32_BLOB_NAME);
    const int score_32_index = engine.getBindingIndex(OUTPUT_SCORE_32_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[input_index], batch_size * INPUT_SIZE * sizeof(float)));

    CHECK(cudaMalloc(&buffers[bbox_8_index], batch_size * OUTPUT_BBOX_8_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[kps_8_index], batch_size * OUTPUT_KPS_8_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[score_8_index], batch_size * OUTPUT_SCORE_8_SIZE * sizeof(float)));

    CHECK(cudaMalloc(&buffers[bbox_16_index], batch_size * OUTPUT_BBOX_16_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[kps_16_index], batch_size * OUTPUT_KPS_16_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[score_16_index], batch_size * OUTPUT_SCORE_16_SIZE * sizeof(float)));

    CHECK(cudaMalloc(&buffers[bbox_32_index], batch_size * OUTPUT_BBOX_32_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[kps_32_index], batch_size * OUTPUT_KPS_32_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[score_32_index], batch_size * OUTPUT_SCORE_32_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[input_index], input, batch_size * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batch_size, buffers, stream, nullptr);

    CHECK(cudaMemcpyAsync(bbox_8, buffers[bbox_8_index], batch_size * OUTPUT_BBOX_8_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(score_8, buffers[score_8_index], batch_size * OUTPUT_SCORE_8_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(kps_8, buffers[kps_8_index], batch_size * OUTPUT_KPS_8_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));

    CHECK(cudaMemcpyAsync(bbox_16, buffers[bbox_16_index], batch_size * OUTPUT_BBOX_16_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(score_16, buffers[score_16_index], batch_size * OUTPUT_SCORE_16_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(kps_16, buffers[kps_16_index], batch_size * OUTPUT_KPS_16_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));

    CHECK(cudaMemcpyAsync(bbox_32, buffers[bbox_32_index], batch_size * OUTPUT_BBOX_32_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(score_32, buffers[score_32_index], batch_size * OUTPUT_SCORE_32_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(kps_32, buffers[kps_32_index], batch_size * OUTPUT_KPS_32_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
   
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[input_index]));

    CHECK(cudaFree(buffers[bbox_8_index]));
    CHECK(cudaFree(buffers[score_8_index]));
    CHECK(cudaFree(buffers[kps_8_index]));

    CHECK(cudaFree(buffers[bbox_16_index]));
    CHECK(cudaFree(buffers[score_16_index]));
    CHECK(cudaFree(buffers[kps_16_index]));

    CHECK(cudaFree(buffers[bbox_32_index]));
    CHECK(cudaFree(buffers[score_32_index]));
    CHECK(cudaFree(buffers[kps_32_index]));
}

cv::Mat resizeImage(cv::Mat image, float scale) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size((image.cols * scale), (image.rows * scale)), 0, 0, cv::INTER_LINEAR);
    return resized_image;
}


int SCRFD_TRT::detect(cv::Mat image, std::vector<FaceObject>& faceobjects, float prob_threshold, float nms_threshold) {
    int width = image.cols;
    int height = image.rows;
    float scale = (float) INPUT_H / std::max(image.cols, image.rows);

    // resize + pad
    cv::Mat pr_image = resizeImage(image, scale);

    int wpad = INPUT_W - pr_image.cols;
    if (wpad > 0) {
        cv::Mat background = cv::Mat(pr_image.rows, wpad, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::hconcat(background, pr_image, pr_image);
    }

    int hpad = INPUT_H - pr_image.rows;
    if (hpad > 0) {
        cv::Mat background = cv::Mat(hpad, INPUT_W, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::vconcat(background, pr_image, pr_image);
    }

    // normalize
    float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = ((float)pr_image.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
        data[i + INPUT_H * INPUT_W] = ((float)pr_image.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        data[i + 2 * INPUT_H * INPUT_W] = ((float)pr_image.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
    }

    // forward
    float score_8[BATCH_SIZE * OUTPUT_SCORE_8_SIZE]; 
    float bbox_8[BATCH_SIZE * OUTPUT_BBOX_8_SIZE]; 
    float kps_8[BATCH_SIZE * OUTPUT_KPS_8_SIZE]; 

    float score_16[BATCH_SIZE * OUTPUT_SCORE_16_SIZE];
    float bbox_16[BATCH_SIZE * OUTPUT_BBOX_16_SIZE];
    float kps_16[BATCH_SIZE * OUTPUT_KPS_16_SIZE]; 

    float score_32[BATCH_SIZE * OUTPUT_SCORE_32_SIZE];
    float bbox_32[BATCH_SIZE * OUTPUT_BBOX_32_SIZE];
    float kps_32[BATCH_SIZE * OUTPUT_KPS_32_SIZE];

    doInference(*context, data, score_8, bbox_8, kps_8, 
                                score_16, bbox_16, kps_16, 
                                score_32, bbox_32, kps_32, BATCH_SIZE);

    // post process
    std::vector<FaceObject> faceproposals;
    
    // stride 8
    int feat_stride = 8;
    std::vector<FaceObject> faceobjects8;
    generate_proposals(anchors_stride_8, num_anchors_stride_8, feat_stride, score_8, bbox_8, kps_8, prob_threshold, faceobjects8, OUTPUT_SCORE_8_SIZE);
    faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    
    // stride 16
    feat_stride = 16;
    std::vector<FaceObject> faceobjects16;
    generate_proposals(anchors_stride_16, num_anchors_stride_16, feat_stride, score_16, bbox_16, kps_16, prob_threshold, faceobjects16, OUTPUT_SCORE_16_SIZE);
    faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());

    // stride 32
    feat_stride = 32;
    std::vector<FaceObject> faceobjects32;
    generate_proposals(anchors_stride_32, num_anchors_stride_32, feat_stride, score_32, bbox_32, kps_32, prob_threshold, faceobjects32, OUTPUT_SCORE_32_SIZE);
    faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    
    // apply nms with nms_threshold
    qsort_descent_inplace(faceproposals);  
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);

    int face_count = picked.size();
    faceobjects.resize(face_count);

    for (int i = 0; i < face_count; i++) {
        faceobjects[i] = faceproposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (faceobjects[i].rect.x - wpad) / scale;
        float y0 = (faceobjects[i].rect.y - hpad) / scale;
        float x1 = (faceobjects[i].rect.x + faceobjects[i].rect.width - wpad) / scale;
        float y1 = (faceobjects[i].rect.y + faceobjects[i].rect.height - hpad) / scale;

        x0 = std::max(std::min(x0, (float)width - 1), 0.f);
        y0 = std::max(std::min(y0, (float)height - 1), 0.f);
        x1 = std::max(std::min(x1, (float)width - 1), 0.f);
        y1 = std::max(std::min(y1, (float)height - 1), 0.f);

        faceobjects[i].rect.x = x0;
        faceobjects[i].rect.y = y0;
        faceobjects[i].rect.width = x1 - x0;
        faceobjects[i].rect.height = y1 - y0;

        {
            float x0 = (faceobjects[i].landmark[0].x - wpad) / scale;
            float y0 = (faceobjects[i].landmark[0].y - hpad) / scale;
            float x1 = (faceobjects[i].landmark[1].x - wpad) / scale;
            float y1 = (faceobjects[i].landmark[1].y - hpad) / scale;
            float x2 = (faceobjects[i].landmark[2].x - wpad) / scale;
            float y2 = (faceobjects[i].landmark[2].y - hpad) / scale;
            float x3 = (faceobjects[i].landmark[3].x - wpad) / scale;
            float y3 = (faceobjects[i].landmark[3].y - hpad) / scale;
            float x4 = (faceobjects[i].landmark[4].x - wpad) / scale;
            float y4 = (faceobjects[i].landmark[4].y - hpad) / scale;

            faceobjects[i].landmark[0].x = std::max(std::min(x0, (float)width - 1), 0.f);
            faceobjects[i].landmark[0].y = std::max(std::min(y0, (float)height - 1), 0.f);
            faceobjects[i].landmark[1].x = std::max(std::min(x1, (float)width - 1), 0.f);
            faceobjects[i].landmark[1].y = std::max(std::min(y1, (float)height - 1), 0.f);
            faceobjects[i].landmark[2].x = std::max(std::min(x2, (float)width - 1), 0.f);
            faceobjects[i].landmark[2].y = std::max(std::min(y2, (float)height - 1), 0.f);
            faceobjects[i].landmark[3].x = std::max(std::min(x3, (float)width - 1), 0.f);
            faceobjects[i].landmark[3].y = std::max(std::min(y3, (float)height - 1), 0.f);
            faceobjects[i].landmark[4].x = std::max(std::min(x4, (float)width - 1), 0.f);
            faceobjects[i].landmark[4].y = std::max(std::min(y4, (float)height - 1), 0.f);
        }
    }
    return 0;
}


float** SCRFD_TRT::generate_anchors(int base_size, const float ratios[], const float scales[]) {
    int num_ratio = sizeof(ratios) / sizeof(ratios[0]);
    int num_scale = sizeof(scales) / sizeof(scales[0]);

    float** anchors = new float*[num_ratio * num_scale];

    const float cx = 0;
    const float cy = 0;

    for (int i = 0; i < num_ratio; i++) {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar); 

        for (int j = 0; j < num_scale; j++) {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            anchors[i * num_scale + j] = new float[4];

            anchors[i * num_scale + j][0] = cx - rs_w * 0.5f;
            anchors[i * num_scale + j][1] = cy - rs_h * 0.5f;
            anchors[i * num_scale + j][2] = cx + rs_w * 0.5f;
            anchors[i * num_scale + j][3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

void SCRFD_TRT::generate_proposals(float** anchors, int num_anchors, int feat_stride, float* score_blob, float* bbox_blob, float* kps_blob, float prob_threshold, std::vector<FaceObject>& faceobjects, int output_score_size) {
    int w = sqrt(output_score_size / 2);
    int h = w;

    // generate face proposal from bbox deltas and shifted anchors
    for (int q = 0; q < num_anchors; q++) {
        const float* anchor = anchors[q];

        // shifted anchor
        float anchor_y = anchor[1];
        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i = 0; i < h; i++) {
            float anchor_x = anchor[0];

            for (int j = 0; j < w; j++) {
                float prob = score_blob[q*w*h + i*w + j];
                
                if (prob >= prob_threshold) {
                    float dx = bbox_blob[4*q*w*h + i*w + j] * feat_stride;
                    float dy = bbox_blob[(4*q + 1)*w*h + i*w + j] * feat_stride;
                    float dw = bbox_blob[(4*q + 2)*w*h + i*w + j] * feat_stride;
                    float dh = bbox_blob[(4*q + 3)*w*h + i*w + j] * feat_stride;

                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float x0 = cx - dx;
                    float y0 = cy - dy;
                    float x1 = cx + dw;
                    float y1 = cy + dh;

                    FaceObject obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.prob = prob;

                    obj.landmark[0].x = cx + kps_blob[8*q*w*h + i*w + j] * feat_stride;
                    obj.landmark[0].y = cy + kps_blob[(8*q + 1)*w*h + i*w + j] * feat_stride;
                    obj.landmark[1].x = cx + kps_blob[(8*q + 2)*w*h + i*w + j] * feat_stride;
                    obj.landmark[1].y = cy + kps_blob[(8*q + 3)*w*h + i*w + j] * feat_stride;
                    obj.landmark[2].x = cx + kps_blob[(8*q + 4)*w*h + i*w + j] * feat_stride;
                    obj.landmark[2].y = cy + kps_blob[(8*q + 5)*w*h + i*w + j] * feat_stride;
                    obj.landmark[3].x = cx + kps_blob[(8*q + 6)*w*h + i*w + j] * feat_stride;
                    obj.landmark[3].y = cy + kps_blob[(8*q + 7)*w*h + i*w + j] * feat_stride;
                    obj.landmark[4].x = cx + kps_blob[(8*q + 8)*w*h + i*w + j] * feat_stride;
                    obj.landmark[4].y = cy + kps_blob[(8*q + 9)*w*h + i*w + j] * feat_stride;

                    faceobjects.push_back(obj);
                }
                anchor_x += feat_stride;
            }
            anchor_y += feat_stride;
        }
    }
}