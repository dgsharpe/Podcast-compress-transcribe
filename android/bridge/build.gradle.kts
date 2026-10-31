// :bridge — "Mimi Bridge": lets any audio app play `.mimi` files by exposing them as seekable,
// decode-on-read WAV content URIs (StorageManager.openProxyFileDescriptor). No root, no app changes.
plugins {
    id("com.android.application")
}

android {
    namespace = "org.joshjob.mimi.bridge"
    compileSdk = 36

    defaultConfig {
        applicationId = "org.joshjob.mimi.bridge"
        minSdk = 26  // openProxyFileDescriptor
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    packaging {
        jniLibs {
            useLegacyPackaging = true
            // NPU graph ships precompiled (mimi_dec_tx_npu.tflite): no on-device QNN compiler needed
            excludes += "**/libLiteRtCompilerPlugin_Qualcomm.so"
        }
    }
    androidResources {
        noCompress += listOf("tflite", "bin")
    }
}

dependencies {
    implementation(project(":mimi"))
    implementation("androidx.annotation:annotation:1.9.1")
}
