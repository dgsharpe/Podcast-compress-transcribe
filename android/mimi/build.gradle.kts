// :mimi — Media3 extension for `.mimi` (Mimi neural codec) audio: extractor + decoder renderer,
// with the native decoder (phone/libmimi) on LiteRT (NPU transformer, GPU conv decoder).
plugins {
    id("com.android.library")
}

val media3 = "1.11.1"   // pocket-casts-android's version
val litert = "2.3.2"

android {
    namespace = "org.joshjob.mimi"
    compileSdk = 36
    ndkVersion = "29.0.14206865"

    defaultConfig {
        minSdk = 24
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { arguments += "-DANDROID_STL=none" } }
        consumerProguardFiles("consumer-rules.pro")
    }
    externalNativeBuild {
        cmake { path = file("src/main/cpp/CMakeLists.txt") }
    }
    // pocket-casts-android's root build adds BuildConfig fields to every module
    buildFeatures { buildConfig = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources {
        // models are copied out of the APK on first use; keep them uncompressed for a fast copy
        noCompress += listOf("tflite", "bin")
    }
    packaging {
        // QNN loads the Hexagon skel (and LiteRT its accelerators) from files on disk
        jniLibs { useLegacyPackaging = true }
    }
}

dependencies {
    api("androidx.media3:media3-exoplayer:$media3")
    api("androidx.media3:media3-extractor:$media3")
    api("androidx.media3:media3-decoder:$media3")
    implementation("androidx.annotation:annotation:1.9.1")
    implementation("com.google.ai.edge.litert:litert:$litert")
    implementation("com.google.ai.edge.litert:litert-gpu:$litert")
    implementation("com.google.ai.edge.litert:litert-npu-runtime-qualcomm:$litert")
}
