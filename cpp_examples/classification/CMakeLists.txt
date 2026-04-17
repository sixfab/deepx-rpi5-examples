# Classification demos — each binary links against deepx_common
# via the add_demo() macro defined in the parent CMakeLists.txt.

add_demo(yolo26cls_demo    yolo26cls_demo.cpp)
add_demo(efficientnet_demo efficientnet_demo.cpp)
add_demo(mobilenet_demo    mobilenet_demo.cpp)
