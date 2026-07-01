package com.example.aiemotiondetector

// appV1.0 Rev 8 (EmotionDetector.kt)
// Rev 8: logit-space bias correction untuk FANE → real-world distribusi shift
//        Model FANE over-predict SAD pada foto real karena dataset mismatch
//        Fix sementara sampai FER2013 model selesai training
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.ln

data class EmotionResult(
    val label: String,
    val confidence: Float,
    val allScores: FloatArray,
    val rawScores: FloatArray
)

class EmotionDetector(context: Context) {

    private val interpreter: Interpreter
    val labels = listOf("ANGRY", "HAPPY", "SAD", "NEUTRAL")

    // Logit-space bias: kurangi probabilitas SAD yang over-dominant di foto real-world
    // Nilai negatif = kurangi (SAD -3.0), positif = naikkan
    // Hapus/set semua ke 0f untuk lihat raw model output
    private val logitBias = floatArrayOf(0.5f, 0.5f, -3.0f, 0.5f)

    init {
        val modelBuffer = loadModelFile(context)
        val options = Interpreter.Options().apply { setNumThreads(2) }
        interpreter = Interpreter(modelBuffer, options)
    }

    private fun loadModelFile(context: Context): ByteBuffer {
        val assetFD = context.assets.openFd("model_emosi_android.tflite")
        val inputStream = FileInputStream(assetFD.fileDescriptor)
        val fileChannel = inputStream.channel
        val mapped = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFD.startOffset,
            assetFD.declaredLength
        )
        inputStream.close()
        return mapped
    }

    fun detectEmotion(bitmap: Bitmap): EmotionResult {
        val argbBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888)
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        else bitmap

        val resized = Bitmap.createScaledBitmap(argbBitmap, 224, 224, true)
        val byteBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(224 * 224)
        resized.getPixels(intValues, 0, 224, 0, 0, 224, 224)

        // Normalisasi ResNetV2: pixel 0-255 → -1.0 hingga 1.0
        for (px in intValues) {
            byteBuffer.putFloat(((px shr 16 and 0xFF) / 127.5f) - 1.0f)
            byteBuffer.putFloat(((px shr  8 and 0xFF) / 127.5f) - 1.0f)
            byteBuffer.putFloat(((px        and 0xFF) / 127.5f) - 1.0f)
        }

        val outputBuffer = Array(1) { FloatArray(labels.size) }
        interpreter.run(byteBuffer, outputBuffer)
        val raw = outputBuffer[0]

        // Log raw output — filter logcat: tag "EmotionDetector"
        Log.d("EmotionDetector", "RAW  → ${labels.mapIndexed { i, l -> "$l:${"%.3f".format(raw[i])}" }.joinToString(" ")}")

        // Logit-space bias correction
        // 1. Hitung approximate log-probabilities (inverse softmax)
        val logits = FloatArray(labels.size) { i ->
            ln(raw[i].toDouble().coerceAtLeast(1e-10)).toFloat() + logitBias[i]
        }
        // 2. Numerically-stable softmax
        val maxLogit = logits.max()!!
        val expLogits = FloatArray(labels.size) { i -> exp((logits[i] - maxLogit).toDouble()).toFloat() }
        val sumExp = expLogits.sum()
        val corrected = FloatArray(labels.size) { i -> expLogits[i] / sumExp }

        Log.d("EmotionDetector", "CORR → ${labels.mapIndexed { i, l -> "$l:${"%.3f".format(corrected[i])}" }.joinToString(" ")}")

        // Cari label dengan confidence tertinggi (setelah koreksi)
        var maxIdx = 0
        for (i in 1 until corrected.size) {
            if (corrected[i] > corrected[maxIdx]) maxIdx = i
        }

        return EmotionResult(
            label      = labels[maxIdx],
            confidence = (corrected[maxIdx] * 100f).coerceIn(0f, 100f),
            allScores  = corrected,
            rawScores  = raw.copyOf()
        )
    }

    fun close() = interpreter.close()
}
