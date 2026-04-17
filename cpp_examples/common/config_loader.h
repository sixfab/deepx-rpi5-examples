#pragma once

#include <string>
#include <utility>
#include <vector>
#include <opencv2/core.hpp>

struct DemoParams {
    std::string modelPath;
    std::string modelPath2;               // optional second model (e.g. face-emotion 2-stage)
    std::string sourceType   = "video";   // "video" | "camera" | "libcamera" | "rtsp" | "image"
    std::string sourcePath;
    int cameraIndex          = 0;
    int inputWidth           = 640;
    int inputHeight          = 640;
    float confThreshold      = 0.3f;
    float iouThreshold       = 0.45f;
    std::string labelPath;
    std::string windowTitle  = "DeepX Demo";
    bool showFps             = true;

    // Optional: advanced-demo geometry. Normalized [0,1] frame coordinates.
    std::vector<std::vector<cv::Point2f>> regions;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> lines;

    // Optional: per-tile source paths for the multi-channel demo. Parsed from
    // the JSON "channels" array: each element is {"path": ".../video.mp4"}.
    std::vector<std::string> channelSources;
};

// Load demo params from a JSON file and then apply CLI overrides.
// Supported CLI flags: --config, --model, --source, --path, --conf, --iou,
//                      --labels, --window, --camera-index, --no-fps.
// `configPath` is the fallback used when --config is not supplied. Pass an
// empty string to auto-derive from argv[0] (e.g. `yolov8_demo` →
// `configs/yolov8_demo.json`). Missing config file is tolerated.
DemoParams loadConfig(const std::string& configPath,
                      int argc = 0, char** argv = nullptr);

// Pick the JSON config path to load, in priority order:
//   1. `--config <path>` CLI argument
//   2. `fallback` if non-empty
//   3. Auto-derived from the executable basename in argv[0]
//      (searched under ./configs and ../configs, plus the same folders
//      relative to the executable).
std::string resolveConfigPath(int argc, char** argv,
                              const std::string& fallback = "");

// If the given path does not exist, re-resolve relative to the running
// executable's directory (and its parent, to cover build/ layouts).
std::string resolveDataPath(const std::string& path);
