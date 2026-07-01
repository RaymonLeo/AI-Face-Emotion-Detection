package com.example.aiemotiondetector

// appV1.0 Rev 6 (EmotionDetector.kt)
// Rev 5: Hapus TFLite Support Library, pakai AssetManager + Bitmap API murni.
// Rev 6: Backend diganti ke LiteRT 1.0.1 (lewat build.gradle.kts) karena
// tensorflow-lite:2.16.1 tidak mendukung op FULLY_CONNECTED versi 12 yang
// dipakai model ini. Import tetap org.tensorflow.lite.* (LiteRT backward-compatible).
import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class EmotionDetector(context: Context) {

    private val interpreter: Interpreter
    private val labels = listOf("ANGRY", "HAPPY", "SAD", "NEUTRAL")

    init {
        val modelBuffer = loadModelFile(context)
        val options = Interpreter.Options().apply {
            setNumThreads(2)
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    // Load model langsung dari AssetManager (tanpa FileUtil dari TFLite Support)
    private fun loadModelFile(context: Context): ByteBuffer {
        val assetFD = context.assets.openFd("model_emosi_android.tflite")
        val inputStream = FileInputStream(assetFD.fileDescriptor)
        val fileChannel = inputStream.channel
        val mappedBuffer = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFD.startOffset,
            assetFD.declaredLength
        )
        inputStream.close()
        return mappedBuffer
    }

    fun detectEmotion(bitmap: Bitmap): Pair<String, Float> {
        // Pastikan bitmap ARGB_8888 agar getPixels berfungsi benar di semua device
        val argbBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888) {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            bitmap
        }

        // Resize ke 224x224 menggunakan Android Bitmap API (bukan TFLite Support)
        val resized = Bitmap.createScaledBitmap(argbBitmap, 224, 224, true)

        // Siapkan ByteBuffer: 1 gambar × 224 × 224 piksel × 3 channel (RGB) × 4 byte (float)
        val byteBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(224 * 224)
        resized.getPixels(intValues, 0, 224, 0, 0, 224, 224)

        // Normalisasi ResNetV2: nilai piksel 0-255 → range -1.0 hingga 1.0
        for (pixelValue in intValues) {
            val r = (pixelValue shr 16) and 0xFF
            val g = (pixelValue shr 8) and 0xFF
            val b = pixelValue and 0xFF
            byteBuffer.putFloat((r / 127.5f) - 1.0f)
            byteBuffer.putFloat((g / 127.5f) - 1.0f)
            byteBuffer.putFloat((b / 127.5f) - 1.0f)
        }

        // Jalankan inference TFLite
        val outputBuffer = Array(1) { FloatArray(labels.size) }
        interpreter.run(byteBuffer, outputBuffer)

        // Cari label dengan confidence tertinggi
        val confidences = outputBuffer[0]
        var maxIdx = 0
        var maxConfidence = confidences[0]
        for (i in 1 until confidences.size) {
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i]
                maxIdx = i
            }
        }

        return Pair(labels[maxIdx], (maxConfidence * 100f).coerceIn(0f, 100f))
    }

    fun close() = interpreter.close()
}
