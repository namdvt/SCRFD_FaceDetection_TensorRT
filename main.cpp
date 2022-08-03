#include <fstream>
#include <map>
#include <chrono>
#include "face_detector.h"
#include "scrfd_trt.h"

#define INPUT_H 640
#define INPUT_W 640

int main(int argc, char** argv) {
    std::string trt_path;
    std::string image_path;
    int repeat_times = 1;

    if (argc < 3) {
        throw std::runtime_error("missing input parameters!");
    } else if (argc == 3) {
        trt_path = argv[1];
        image_path = argv[2];
    } else if (argc == 4) {
        trt_path = argv[1];
        image_path = argv[2];
        repeat_times = strtol(argv[3], NULL, 10);
    } else {
        throw std::runtime_error("too many input parameters!");
    }
    
    cv::Mat image = cv::imread(image_path.c_str());
    std::vector<FaceObject> faceobjects;
    SCRFD_TRT* scrfd_trt = new SCRFD_TRT(trt_path);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<repeat_times; i++) {
        scrfd_trt->detect(image, faceobjects, 0.25, 0.4);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "fps: " << float(repeat_times) / duration.count() * 1000 << "\n";
    scrfd_trt->draw(image, faceobjects);

    return 0;
}