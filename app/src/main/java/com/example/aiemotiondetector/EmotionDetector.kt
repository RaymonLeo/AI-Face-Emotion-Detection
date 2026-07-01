package com.example.aiemotiondetector

// appV1.0 Rev 8 (EmotionDetector.kt)
// Rev 8: tambah EmotionResult dengan allScores untuk debug display semua 4 kelas
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

data class EmotionResult(
    val label: String,
    val confidence: Float,
    val allScores: FloatArray
)

class EmotionDetector(context: Context) {

    private val interpreter: Interpreter
    val labels = listOf("ANGRY", "HAPPY", "SAD", "NEUTRAL")

    init {
        val modelBuffer = loadModelFile(context)
        val options = Interpreter.Options().apply {
            setNumThreads(2)
        }
        interpreter = Interpreter(modelBuffer, options)
    }

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

    fun detectEmotion(bitmap: Bitmap): EmotionResult {
        val argbBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888) {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            bitmap
        }

        val resized = Bitmap.createScaledBitmap(argbBitmap, 224, 224, true)
        val byteBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(224 * 224)
        resized.getPixels(intValues, 0, 224, 0, 0, 224, 224)

        // Normalisasi ResNetV2: pixel 0-255 → range -1.0 hingga 1.0
        for (pixelValue in intValues) {
            val r = (pixelValue shr 16) and 0xFF
            val g = (pixelValue shr 8) and 0xFF
            val b = pixelValue and 0xFF
            byteBuffer.putFloat((r / 127.5f) - 1.0f)
            byteBuffer.putFloat((g / 127.5f) - 1.0f)
            byteBuffer.putFloat((b / 127.5f) - 1.0f)
        }

        val outputBuffer = Array(1) { FloatArray(labels.size) }
        interpreter.run(byteBuffer, outputBuffer)

        val confidences = outputBuffer[0]

        // Log semua 4 score untuk diagnosis — filter logcat dengan tag "EmotionDetector"
        Log.d("EmotionDetector", buildString {
            append("RAW SCORES → ")
            labels.forEachIndexed { i, lbl -> append("$lbl: ${"%.4f".format(confidences[i])}  ") }
        })

        var maxIdx = 0
        var maxConfidence = confidences[0]
        for (i in 1 until confidences.size) {
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i]
                maxIdx = i
            }
        }

        return EmotionResult(
            label = labels[maxIdx],
            confidence = (maxConfidence * 100f).coerceIn(0f, 100f),
            allScores = confidences.copyOf()
        )
    }

    fun close() = interpreter.close()
}
