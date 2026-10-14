cmake_minimum_required(VERSION 3.16)
project(bibit_voice VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BIBIT_SHERPA_ROOT
    "/opt/bibit/models/runtime/sherpa-onnx-v1.12.36-linux-riscv64-spacemit-shared"
    CACHE PATH "sherpa-onnx SpacemiT runtime root")
set(BIBIT_SHERPA_INCLUDE "/opt/bibit/include" CACHE PATH "sherpa-onnx include root")

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)

add_executable(bibit-audio
    src/audio/main.cpp
    src/audio/config.cpp
    src/audio/config.h
    src/audio/resampler.cpp
    src/audio/resampler.h
    src/audio/wav.cpp
    src/audio/wav.h
    src/audio/pipewire_capture.cpp
    src/audio/pipewire_capture.h
    src/audio/qwen3_asr.cpp
    src/audio/qwen3_asr.h
    src/audio/pipeline.cpp
    src/audio/pipeline.h
    src/audio/socket_server.cpp
    src/audio/socket_server.h
)
target_include_directories(bibit-audio PRIVATE
    ${BIBIT_SHERPA_INCLUDE}
    ${PIPEWIRE_INCLUDE_DIRS}
)
target_link_directories(bibit-audio PRIVATE
    "${BIBIT_SHERPA_ROOT}/lib"
    ${PIPEWIRE_LIBRARY_DIRS}
)
target_compile_options(bibit-audio PRIVATE ${PIPEWIRE_CFLAGS_OTHER})
target_link_libraries(bibit-audio PRIVATE
    sherpa-onnx-c-api
    ${PIPEWIRE_LIBRARIES}
    Threads::Threads
    dl
)
set_target_properties(bibit-audio PROPERTIES
    BUILD_RPATH "${BIBIT_SHERPA_ROOT}/lib"
)

install(TARGETS bibit-audio RUNTIME DESTINATION bin)
install(PROGRAMS src/tool/bibit_voice_tool.py
    DESTINATION bin
    RENAME bibit-voice-tool
)
