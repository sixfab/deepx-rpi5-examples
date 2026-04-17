# Object detection demos — each binary links against deepx_common
# via the add_demo() macro defined in the parent CMakeLists.txt.

add_demo(yolov8_demo          yolov8_demo.cpp)
add_demo(yolov5_demo          yolov5_demo.cpp)
add_demo(yolov11_demo         yolov11_demo.cpp)
add_demo(ppe_detection_demo   ppe_detection_demo.cpp)
add_demo(mask_detection_demo  mask_detection_demo.cpp)
