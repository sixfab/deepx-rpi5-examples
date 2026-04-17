#pragma once

#include <memory>
#include <string>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>

enum class SourceType { VIDEO, CAMERA, LIBCAMERA, RTSP, IMAGE };

SourceType parseSourceType(const std::string& typeStr);
std::string sourceTypeToString(SourceType t);

class InputSource {
public:
    // path: filesystem path for VIDEO / IMAGE / RTSP URL;
    //       "idx:W:H:fps" colon-form for LIBCAMERA; ignored for CAMERA.
    // cameraIndex: /dev/videoN index used when type == CAMERA.
    explicit InputSource(SourceType type,
                         const std::string& path = "",
                         int cameraIndex = 0);
    ~InputSource();

    // Returns false once the stream is exhausted (non-looping sources) or
    // a read error occurs. For VIDEO sources the stream loops automatically.
    bool read(cv::Mat& frame);

    // Restart a VIDEO source from the beginning. No-op for other types.
    void rewind();

    void release();
    bool isOpened() const;

    SourceType type() const { return m_type; }

private:
    bool openLibcamera(const std::string& spec);

    SourceType m_type;
    std::string m_path;
    int m_cameraIndex;

    cv::VideoCapture m_cap;
    cv::Mat m_still;   // for IMAGE sources
    bool m_eof = false;
};
