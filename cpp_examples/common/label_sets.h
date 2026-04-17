#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace labels {

// COCO 80-class detection labels (YOLO family default).
inline const std::vector<std::string> COCO80 = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
};

// PPE-detection classes (ppe-detection-v11 model).
inline const std::vector<std::string> PPE_CLASSES = {
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
};

// Mask-detection classes (mask_detection_yolov5_ppu model).
inline const std::vector<std::string> MASK_CLASSES = {
    "mask", "no_mask",
};

// Cityscapes 19-class palette — used by DeepLabV3+ semantic segmentation.
// Returned as OpenCV BGR Scalars (the upstream palette is defined as RGB, we
// swap channels so OpenCV draws them correctly).
inline std::vector<cv::Scalar> cityscapesPalette() {
    // (R, G, B) tuples from the canonical Cityscapes palette, in class-index order:
    //   0 road, 1 sidewalk, 2 building, 3 wall, 4 fence, 5 pole, 6 traffic-light,
    //   7 traffic-sign, 8 vegetation, 9 terrain, 10 sky, 11 person, 12 rider,
    //   13 car, 14 truck, 15 bus, 16 train, 17 motorcycle, 18 bicycle.
    static const int kRgb[19][3] = {
        {128, 64, 128},  {244, 35, 232},  {70, 70, 70},    {102, 102, 156},
        {190, 153, 153}, {153, 153, 153}, {250, 170, 30},  {220, 220, 0},
        {107, 142, 35},  {152, 251, 152}, {70, 130, 180},  {220, 20, 60},
        {255, 0, 0},     {0, 0, 142},     {0, 0, 70},      {0, 60, 100},
        {0, 80, 100},    {0, 0, 230},     {119, 11, 32},
    };
    std::vector<cv::Scalar> out;
    out.reserve(19);
    for (const auto& c : kRgb) {
        out.emplace_back(c[2], c[1], c[0]);  // BGR
    }
    return out;
}

// Read one label per line from a file. Returns an empty vector when the
// file is missing or unreadable.
inline std::vector<std::string> loadFromFile(const std::string& path) {
    std::vector<std::string> out;
    if (path.empty() || !std::filesystem::exists(path)) return out;
    std::ifstream in(path);
    if (!in.is_open()) return out;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() &&
               (line.back() == '\r' || line.back() == '\n' ||
                line.back() == ' '  || line.back() == '\t')) {
            line.pop_back();
        }
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

// ImageNet-1k labels. Attempts to lazy-load a labels file from a few well-known
// locations; falls back to "class_N" strings when nothing is available so the
// classification demos still display output without a labels file.
inline std::vector<std::string> imagenet1000() {
    static const std::vector<std::string> kCandidates = {
        "../models/imagenet1000.txt",
        "../labels/imagenet1000.txt",
        "../configs/imagenet1000.txt",
        "./imagenet1000.txt",
    };
    for (const auto& p : kCandidates) {
        auto labs = loadFromFile(p);
        if (labs.size() >= 1000) return labs;
    }
    std::vector<std::string> out;
    out.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        out.push_back("class_" + std::to_string(i));
    }
    return out;
}

} // namespace labels
