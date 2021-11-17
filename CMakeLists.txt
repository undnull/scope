cmake_minimum_required(VERSION 3.10)
project(scope LANGUAGES C VERSION 0.0.1)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(PA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(glad)
add_subdirectory(glfw)
add_subdirectory(portaudio)

add_executable(scope "${CMAKE_CURRENT_LIST_DIR}/scope.c")
target_compile_definitions(scope PRIVATE GLFW_INCLUDE_NONE)
target_link_libraries(scope PRIVATE glad glfw PortAudio)
