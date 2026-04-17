# Segmentation demos — each binary links against deepx_common
# via the add_demo() macro defined in the parent CMakeLists.txt.

add_demo(yolov8seg_demo  yolov8seg_demo.cpp)
add_demo(yolo26seg_demo  yolo26seg_demo.cpp)
add_demo(deeplabv3_demo  deeplabv3_demo.cpp)
