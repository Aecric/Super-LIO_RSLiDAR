cmake_minimum_required(VERSION 3.16)
project(super_lio VERSION 2.0 LANGUAGES CXX C)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE STRING "Suppress developer warnings" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pthread -fexceptions -flto=auto")
set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -fPIC")
add_definitions(-DROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/\")


find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(super_lio_loop_msgs REQUIRED)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(PCL REQUIRED)
find_package(Glog REQUIRED)
find_package(basic REQUIRED)
find_package(TBB REQUIRED)

# Fixed-map relocalization: KISS-Matcher (prior-free global registration) is
# built from the vendored source; its ROBIN dependency is fetched at configure
# time. Never pick up a robin/kiss_matcher left in a shared install prefix by
# another package (lightning-lm / hikari_loclite vendor the same libraries).
if(NOT TARGET kiss_matcher::kiss_matcher_core)
  set(USE_SYSTEM_TBB ON CACHE BOOL "Use system TBB for KISS-Matcher" FORCE)
  set(USE_SYSTEM_ROBIN OFF CACHE BOOL "Fetch ROBIN for KISS-Matcher" FORCE)
  # Static KISS/ROBIN archives end up inside the shared liblio.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/KISS-Matcher/cpp/kiss_matcher
    ${CMAKE_BINARY_DIR}/_deps/kiss_matcher-build
    EXCLUDE_FROM_ALL)
endif()


rosidl_generate_interfaces(${PROJECT_NAME}
  msg/CloudPose.msg
  msg/CloudPose2.msg
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)

rosidl_get_typesupport_target(
  cpp_typesupport_target
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp"
)

include_directories(
  include
  3rdparty
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Glog_INCLUDE_DIRS}
)

file(GLOB_RECURSE LIO_LIB_SRC "src/lio/*.cpp" "src/ros/*.cpp")
file(GLOB_RECURSE LIO_LIB_HEADERS "include/lio/*.hpp" "include/lio/*.h")

# GenZ-ICP core (refinement against the fixed map), vendored with Sophus and tsl.
set(GENZ_ICP_SRC
  thirdparty/genz_icp/genz_icp/VoxelHashMap.cpp
  thirdparty/genz_icp/genz_icp/Registration.cpp
)

add_library(lio ${LIO_LIB_SRC} ${LIO_LIB_HEADERS} ${GENZ_ICP_SRC})
target_include_directories(lio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/genz_icp)

target_link_libraries(lio PUBLIC
  Eigen3::Eigen
  ${PCL_LIBRARIES}
  rclcpp::rclcpp
  ${std_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${pcl_ros_TARGETS}
  ${livox_ros_driver2_TARGETS}
  ${super_lio_loop_msgs_TARGETS}
  basic::basic
  glog
  pthread
  ${cpp_typesupport_target}
  PRIVATE
  kiss_matcher::kiss_matcher_core
  TBB::tbb
)

add_executable(super_lio_node src/apps/super_lio_node.cpp)
target_link_libraries(super_lio_node
  lio
  ${cpp_typesupport_target}
)

add_executable(relocation_node src/apps/relocation_node.cpp)
target_link_libraries(relocation_node
  lio
  ${cpp_typesupport_target}
)

install(TARGETS
  lio
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(TARGETS
  super_lio_node
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(run_loclite_online src/apps/run_loclite_online.cpp)
target_link_libraries(run_loclite_online lio ${cpp_typesupport_target})

install(TARGETS
  run_loclite_online
  relocation_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  3rdparty/
  DESTINATION include/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

# KISS-Matcher and its fetched ROBIN/PMC/xenium carry their own install rules,
# which would put static libs, headers and CMake exports into the prefix and
# collide in dpkg with lightning-lm / hikari_loclite shipping the same files.
# Everything is linked statically, so drop them after the subtree installed.
# DESTDIR-aware, multiarch-libdir-aware, exact paths only.
include(GNUInstallDirs)
string(CONFIGURE [[
  set(_super_lio_strip_root "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
  file(REMOVE_RECURSE
    "${_super_lio_strip_root}/@CMAKE_INSTALL_INCLUDEDIR@/kiss_matcher"
    "${_super_lio_strip_root}/@CMAKE_INSTALL_INCLUDEDIR@/robin"
    "${_super_lio_strip_root}/@CMAKE_INSTALL_INCLUDEDIR@/pmc"
    "${_super_lio_strip_root}/@CMAKE_INSTALL_INCLUDEDIR@/xenium"
    "${_super_lio_strip_root}/@CMAKE_INSTALL_LIBDIR@/cmake/kiss_matcher"
    "${_super_lio_strip_root}/@CMAKE_INSTALL_LIBDIR@/cmake/robin"
    "${_super_lio_strip_root}/@CMAKE_INSTALL_LIBDIR@/cmake/pmc"
    "${_super_lio_strip_root}/@CMAKE_INSTALL_LIBDIR@/cmake/xenium"
    "${_super_lio_strip_root}/@CMAKE_INSTALL_LIBDIR@/libkiss_matcher_core.a"
    "${_super_lio_strip_root}/@CMAKE_INSTALL_LIBDIR@/librobin.a"
    "${_super_lio_strip_root}/@CMAKE_INSTALL_LIBDIR@/libpmc.a")
]] _super_lio_strip_code @ONLY)
install(CODE "${_super_lio_strip_code}")

ament_package()
