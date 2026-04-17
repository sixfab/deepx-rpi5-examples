#include "config_loader.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

template <typename T>
void getIfPresent(const json& j, const char* key, T& dst) {
    auto it = j.find(key);
    if (it != j.end() && !it->is_null()) {
        try { dst = it->get<T>(); } catch (...) { /* keep default */ }
    }
}

void parseRegions(const json& j, DemoParams& p) {
    auto it = j.find("regions");
    if (it == j.end() || !it->is_array()) return;
    for (const auto& poly : *it) {
        if (!poly.is_array()) continue;
        std::vector<cv::Point2f> pts;
        for (const auto& pt : poly) {
            if (pt.is_array() && pt.size() >= 2) {
                pts.emplace_back(pt[0].get<float>(), pt[1].get<float>());
            }
        }
        if (!pts.empty()) p.regions.push_back(std::move(pts));
    }
}

void parseLines(const json& j, DemoParams& p) {
    auto it = j.find("lines");
    if (it == j.end() || !it->is_array()) return;
    for (const auto& line : *it) {
        if (!line.is_array() || line.size() < 2) continue;
        if (!line[0].is_array() || !line[1].is_array()) continue;
        cv::Point2f a(line[0][0].get<float>(), line[0][1].get<float>());
        cv::Point2f b(line[1][0].get<float>(), line[1][1].get<float>());
        p.lines.emplace_back(a, b);
    }
}

void parseChannels(const json& j, DemoParams& p) {
    auto it = j.find("channels");
    if (it == j.end() || !it->is_array()) return;
    for (const auto& c : *it) {
        if (c.is_object()) {
            auto pit = c.find("path");
            if (pit != c.end() && pit->is_string()) {
                p.channelSources.push_back(pit->get<std::string>());
            }
        } else if (c.is_string()) {
            p.channelSources.push_back(c.get<std::string>());
        }
    }
}

std::string parseConfigFromCli(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--config" || a == "-c") return argv[i + 1];
    }
    return "";
}

std::string deriveConfigFromExe(int argc, char** argv) {
    if (argc < 1 || !argv || !argv[0] || !argv[0][0]) return "";
    fs::path exe(argv[0]);
    const std::string stem = exe.stem().string();
    if (stem.empty()) return "";

    std::vector<fs::path> candidates = {
        fs::path("configs")        / (stem + ".json"),
        fs::path("..") / "configs" / (stem + ".json"),
    };

    std::error_code ec;
    fs::path exeAbs = fs::absolute(exe, ec);
    if (!ec) {
        fs::path exeDir = exeAbs.parent_path();
        candidates.push_back(exeDir / "configs"        / (stem + ".json"));
        candidates.push_back(exeDir / ".." / "configs" / (stem + ".json"));
    }

    for (const auto& c : candidates) {
        if (fs::exists(c)) return c.lexically_normal().string();
    }
    // Return the conventional location so downstream error messages are useful.
    return (fs::path("..") / "configs" / (stem + ".json")).string();
}

} // namespace

std::string resolveConfigPath(int argc, char** argv,
                              const std::string& fallback) {
    std::string cli = parseConfigFromCli(argc, argv);
    if (!cli.empty()) return cli;
    if (!fallback.empty()) return fallback;
    return deriveConfigFromExe(argc, argv);
}

DemoParams loadConfig(const std::string& configPath, int argc, char** argv) {
    DemoParams p;

    const std::string resolved = resolveConfigPath(argc, argv, configPath);

    // 1. JSON file
    if (!resolved.empty() && fs::exists(resolved)) {
        try {
            std::ifstream in(resolved);
            json j;
            in >> j;
            getIfPresent(j, "model_path",     p.modelPath);
            getIfPresent(j, "model_path_2",   p.modelPath2);
            getIfPresent(j, "source_type",    p.sourceType);
            getIfPresent(j, "source_path",    p.sourcePath);
            getIfPresent(j, "camera_index",   p.cameraIndex);
            getIfPresent(j, "input_width",    p.inputWidth);
            getIfPresent(j, "input_height",   p.inputHeight);
            getIfPresent(j, "conf_threshold", p.confThreshold);
            getIfPresent(j, "iou_threshold",  p.iouThreshold);
            getIfPresent(j, "label_path",     p.labelPath);
            getIfPresent(j, "window_title",   p.windowTitle);
            getIfPresent(j, "show_fps",       p.showFps);
            parseRegions(j, p);
            parseLines(j, p);
            parseChannels(j, p);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[WARN] Failed to parse %s: %s (using defaults)\n",
                         resolved.c_str(), e.what());
        }
    } else if (!resolved.empty()) {
        std::fprintf(stderr, "[INFO] Config file not found: %s (using defaults)\n",
                     resolved.c_str());
    }

    // 2. CLI overrides
    auto valueFor = [&](int i) -> const char* {
        return (i + 1 < argc) ? argv[i + 1] : nullptr;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--config" || a == "-c") && valueFor(i)) { ++i; /* consumed */ }
        else if (a == "--model"         && valueFor(i)) { p.modelPath   = valueFor(i); ++i; }
        else if ((a == "--model2" || a == "--reid-model") && valueFor(i))
                                                   { p.modelPath2  = valueFor(i); ++i; }
        else if (a == "--source"   && valueFor(i)) { p.sourceType  = valueFor(i); ++i; }
        else if (a == "--path"     && valueFor(i)) { p.sourcePath  = valueFor(i); ++i; }
        else if (a == "--labels"   && valueFor(i)) { p.labelPath   = valueFor(i); ++i; }
        else if (a == "--window"   && valueFor(i)) { p.windowTitle = valueFor(i); ++i; }
        else if (a == "--conf"     && valueFor(i)) { p.confThreshold = std::stof(valueFor(i)); ++i; }
        else if (a == "--iou"      && valueFor(i)) { p.iouThreshold  = std::stof(valueFor(i)); ++i; }
        else if (a == "--camera-index" && valueFor(i)) { p.cameraIndex = std::stoi(valueFor(i)); ++i; }
        else if (a == "--no-fps")                  { p.showFps = false; }
    }

    // Fix up relative paths so demos can be run from either the repo root
    // or a build/ subdirectory without hitting "file not found".
    p.modelPath  = resolveDataPath(p.modelPath);
    if (!p.modelPath2.empty()) p.modelPath2 = resolveDataPath(p.modelPath2);
    if (p.sourceType == "video" || p.sourceType == "image") {
        p.sourcePath = resolveDataPath(p.sourcePath);
    }
    if (!p.labelPath.empty()) {
        p.labelPath = resolveDataPath(p.labelPath);
    }
    return p;
}

std::string resolveDataPath(const std::string& path) {
    if (path.empty()) return path;
    if (fs::exists(path)) return path;

    // Try relative to the current working directory's parent (covers the
    // `cd build && ./demo` layout used by every example).
    fs::path cwd = fs::current_path();
    fs::path cand = cwd.parent_path() / path;
    if (fs::exists(cand)) return cand.string();

    cand = cwd / path;
    if (fs::exists(cand)) return cand.string();

    return path;  // let the caller's open() fail with a real error
}
