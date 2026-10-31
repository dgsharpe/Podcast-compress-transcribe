plugins {
    id("com.android.application")
}

android {
    namespace = "org.joshjob.mimi.demo"
    compileSdk = 36

    defaultConfig {
        applicationId = "org.joshjob.mimi.demo"
        minSdk = 24
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
        noCompress += listOf("tflite", "bin")  // apps must set this; the library's setting isn't inherited
    }
}

dependencies {
    implementation(project(":mimi"))
    implementation("androidx.media3:media3-ui:1.11.1")
    implementation("androidx.annotation:annotation:1.9.1")
}
