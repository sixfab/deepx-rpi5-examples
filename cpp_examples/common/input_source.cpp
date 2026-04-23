#include "input_source.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <thread>
#include <vector>

SourceType parseSourceType(const std::string& typeStr) {
    std::string s;
    s.reserve(typeStr.size());
    for (char c : typeStr) s.push_back(static_cast<char>(std::tolower(c)));
    if (s == "video")     return SourceType::VIDEO;
    if (s == "camera" || s == "webcam") return SourceType::CAMERA;
    if (s == "libcamera" || s == "rpicam") return SourceType::LIBCAMERA;
    if (s == "rtsp")      return SourceType::RTSP;
    if (s == "image")     return SourceType::IMAGE;
    std::fprintf(stderr, "[WARN] Unknown source type '%s', defaulting to video\n",
                 typeStr.c_str());
    return SourceType::VIDEO;
}

std::string sourceTypeToString(SourceType t) {
    switch (t) {
        case SourceType::VIDEO:     return "video";
        case SourceType::CAMERA:    return "camera";
        case SourceType::LIBCAMERA: return "libcamera";
        case SourceType::RTSP:      return "rtsp";
        case SourceType::IMAGE:     return "image";
    }
    return "video";
}

InputSource::InputSource(SourceType type, const std::string& path, int cameraIndex)
    : m_type(type), m_path(path), m_cameraIndex(cameraIndex)
{
    bool ok = false;
    switch (type) {
        case SourceType::IMAGE:
            m_still = cv::imread(path);
            ok = !m_still.empty();
            if (!ok) {
                std::fprintf(stderr, "[ERROR] Could not read image file: %s\n",
                             path.c_str());
            }
            break;

        case SourceType::CAMERA:
            ok = m_cap.open(cameraIndex);
            if (ok) m_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            else std::fprintf(stderr, "[ERROR] Could not open camera index %d\n",
                              cameraIndex);
            break;

        case SourceType::LIBCAMERA:
            ok = openLibcamera(path);
            break;

        case SourceType::RTSP:
            ok = m_cap.open(path);
            if (!ok) std::fprintf(stderr, "[ERROR] Could not open RTSP: %s\n",
                                  path.c_str());
            break;

        case SourceType::VIDEO:
        default:
            ok = m_cap.open(path);
            if (!ok) std::fprintf(stderr, "[ERROR] Could not open video: %s\n",
                                  path.c_str());
            break;
    }
    if (!ok) m_eof = true;
}

InputSource::~InputSource() {
    release();
}

bool InputSource::openLibcamera(const std::string& spec) {
#ifdef ENABLE_LIBCAMERA
    // Parse "idx:W:H:fps". All fields optional.
    int idx = 0, w = 1536, h = 864, fps = 30;
    std::stringstream ss(spec);
    std::string tok;
    std::vector<std::string> parts;
    while (std::getline(ss, tok, ':')) parts.push_back(tok);
    try {
        if (parts.size() >= 1 && !parts[0].empty()) idx = std::stoi(parts[0]);
        if (parts.size() >= 2 && !parts[1].empty()) w   = std::stoi(parts[1]);
        if (parts.size() >= 3 && !parts[2].empty()) h   = std::stoi(parts[2]);
        if (parts.size() >= 4 && !parts[3].empty()) fps = std::stoi(parts[3]);
    } catch (...) { /* keep defaults */ }

    std::string src = "libcamerasrc";
    if (idx > 0) src += " camera-name=" + std::to_string(idx);

    const std::string pipeline =
        src + " ! "
        "video/x-raw,format=NV12"
        ",width="     + std::to_string(w)   +
        ",height="    + std::to_string(h)   +
        ",framerate=" + std::to_string(fps) + "/1"
        " ! videoconvert"
        " ! video/x-raw,format=BGR"
        " ! appsink max-buffers=1 drop=true sync=false";

    std::printf("[INFO] Opening libcamera: %s\n", pipeline.c_str());
    if (!m_cap.open(pipeline, cv::CAP_GSTREAMER)) {
        std::fprintf(stderr, "[ERROR] libcamera GStreamer pipeline failed to open\n");
        return false;
    }

    // Warm-up: libcamera needs a few frames to negotiate caps.
    for (int i = 0; i < 30; ++i) {
        if (m_cap.grab()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::fprintf(stderr, "[ERROR] libcamera: no frames after warm-up\n");
    m_cap.release();
    return false;
#else
    (void)spec;
    std::fprintf(stderr, "[ERROR] libcamera support not compiled in "
                          "(rebuild with -DENABLE_LIBCAMERA=ON)\n");
    return false;
#endif
}

bool InputSource::read(cv::Mat& frame) {
    if (m_eof) return false;

    if (m_type == SourceType::IMAGE) {
        frame = m_still.clone();
        return !frame.empty();
    }

    if (!m_cap.isOpened()) return false;

    bool ok = m_cap.read(frame);

    // VIDEO sources auto-loop, mirroring the Python demos and the existing
    // VideoSource behaviour.
    if (!ok && m_type == SourceType::VIDEO) {
        m_cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        ok = m_cap.read(frame);
    }
    return ok && !frame.empty();
}

void InputSource::rewind() {
    if (m_type == SourceType::VIDEO && m_cap.isOpened()) {
        m_cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }
}

void InputSource::release() {
    if (m_cap.isOpened()) m_cap.release();
    m_still.release();
}

bool InputSource::isOpened() const {
    if (m_type == SourceType::IMAGE) return !m_still.empty();
    return m_cap.isOpened();
}
