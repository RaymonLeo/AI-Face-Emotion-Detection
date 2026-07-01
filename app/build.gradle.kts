// appV1.0 Rev 6 (build.gradle.kts)
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.aiemotiondetector"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.aiemotiondetector"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx.v1120)
    implementation(libs.material)

    // 1. LiteRT 1.0.1 — nama baru TFLite dari Google, support FULLY_CONNECTED v12+
    //    Menggantikan org.tensorflow:tensorflow-lite:2.16.1 yang tidak support op v12
    implementation(libs.litert)

    // 2. SDK Resmi Gemini AI untuk Android
    implementation(libs.generativeai)

    // 3. ML Kit Face Detection — crop wajah sebelum inference agar sesuai distribusi training
    implementation(libs.mlkit.face.detection)

    // 4. Coroutines untuk pemrosesan Asynchronous
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.androidx.lifecycle.runtime.ktx)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}
