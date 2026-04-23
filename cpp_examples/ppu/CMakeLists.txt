# Educational PPU demos — mirror object_detection demos but log the
# first-frame tensor layout so you can see what the NPU's PPU returns.

add_demo(yolov8_ppu_demo yolov8_ppu_demo.cpp)
add_demo(yolov11_ppu_demo yolov11_ppu_demo.cpp)
add_demo(yolov5_ppu_demo yolov5_ppu_demo.cpp)
