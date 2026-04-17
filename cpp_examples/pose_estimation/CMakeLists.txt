# Pose-estimation demos — each binary links against deepx_common
# via the add_demo() macro defined in the parent CMakeLists.txt.

add_demo(body_pose_demo     body_pose_demo.cpp)
add_demo(hand_landmark_demo hand_landmark_demo.cpp)
