cmake_minimum_required(VERSION 3.22.1)
project(mimi_jni C)

# libmimi sources live in phone/libmimi (shared with the Termux CLI build).
set(LIBMIMI ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../phone/libmimi)
set(LITERT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/litert)

# libLiteRt.so is packaged by the com.google.ai.edge.litert:litert AAR; this copy (fetched by
# fetch_deps.sh) is only for linking.
add_library(litert SHARED IMPORTED)
set_target_properties(litert PROPERTIES IMPORTED_LOCATION ${LITERT}/lib/${ANDROID_ABI}/libLiteRt.so)

add_library(mimi_jni SHARED mimi_jni.c ${LIBMIMI}/mimi.c)
target_include_directories(mimi_jni PRIVATE ${LIBMIMI} ${LITERT}/include)
target_compile_definitions(mimi_jni PRIVATE LITERT_DISABLE_OPENGL_SUPPORT)
target_compile_options(mimi_jni PRIVATE -O3 -Wall -std=c11)
target_link_libraries(mimi_jni litert log m)
target_link_options(mimi_jni PRIVATE -Wl,-z,max-page-size=16384)
