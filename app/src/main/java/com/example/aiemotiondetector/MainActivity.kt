package com.example.aiemotiondetector

// appV1.0 Rev 7 (MainActivity.kt)
// Fix: tambah ML Kit face detection — crop wajah sebelum inference
//      Model FANE ditraining dengan foto wajah yang sudah di-crop ketat,
//      tanpa ini model melihat foto penuh (badan+background) yang tidak sesuai distribusi training
// Fix: tambah EXIF orientation — foto dari kamera sering tersimpan miring 90°
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.google.ai.client.generativeai.GenerativeModel
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.resume
import android.graphics.ImageDecoder

class MainActivity : AppCompatActivity() {

    private var emotionDetector: EmotionDetector? = null
    private var modelInitError: String = ""
    private var currentEmotion: String = ""

    private var cameraImageUri: Uri? = null

    private lateinit var layoutPlaceholder: LinearLayout
    private lateinit var ivPreview: ImageView
    private lateinit var tvEmotionEmoji: TextView
    private lateinit var tvHasilEmosi: TextView
    private lateinit var tvConfidenceLabel: TextView
    private lateinit var progressConfidence: LinearProgressIndicator
    private lateinit var btnPilihFoto: MaterialButton
    private lateinit var btnTanyaGemini: MaterialButton
    private lateinit var tvResponsGemini: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var cardResult: MaterialCardView
    private lateinit var cardGemini: MaterialCardView

    private val pickImageFromGallery =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { processSelectedImage(it) }
        }

    private val takePictureFromCamera =
        registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
            if (success) {
                cameraImageUri?.let { processSelectedImage(it) }
            } else {
                Toast.makeText(this, "Pengambilan foto dibatalkan.", Toast.LENGTH_SHORT).show()
            }
        }

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                openCamera()
            } else {
                Toast.makeText(
                    this,
                    "Izin kamera ditolak. Aktifkan di Pengaturan → Aplikasi.",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        layoutPlaceholder = findViewById(R.id.layoutPlaceholder)
        ivPreview = findViewById(R.id.ivPreview)
        tvEmotionEmoji = findViewById(R.id.tvEmotionEmoji)
        tvHasilEmosi = findViewById(R.id.tvHasilEmosi)
        tvConfidenceLabel = findViewById(R.id.tvConfidenceLabel)
        progressConfidence = findViewById(R.id.progressConfidence)
        btnPilihFoto = findViewById(R.id.btnPilihFoto)
        btnTanyaGemini = findViewById(R.id.btnTanyaGemini)
        tvResponsGemini = findViewById(R.id.tvResponsGemini)
        progressBar = findViewById(R.id.progressBar)
        cardResult = findViewById(R.id.cardResult)
        cardGemini = findViewById(R.id.cardGemini)

        initModel()

        btnPilihFoto.setOnClickListener {
            if (emotionDetector == null) {
                initModel()
            }
            if (emotionDetector == null) {
                showModelError(modelInitError)
                return@setOnClickListener
            }
            showSourceChooserDialog()
        }

        btnTanyaGemini.setOnClickListener {
            mintaNasihatGemini(currentEmotion)
        }
    }

    private fun initModel() {
        if (emotionDetector != null) return
        try {
            emotionDetector = EmotionDetector(this)
        } catch (e: Exception) {
            modelInitError = e.localizedMessage ?: "Unknown error saat load model"
            showModelError(modelInitError)
        }
    }

    private fun showModelError(errorMsg: String) {
        val errorColor = ContextCompat.getColor(this, R.color.error)
        cardResult.setCardBackgroundColor(ColorStateList.valueOf(errorColor))
        tvEmotionEmoji.text = "⚠️"
        tvHasilEmosi.text = "Model Gagal Load"
        tvConfidenceLabel.text = "Error: $errorMsg"
        progressConfidence.progress = 0
    }

    private fun showSourceChooserDialog() {
        val options = arrayOf(
            "📷   Ambil Foto (Kamera)",
            "🖼️   Pilih dari Galeri"
        )
        MaterialAlertDialogBuilder(this)
            .setTitle("Pilih Sumber Foto Wajah")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> checkCameraPermissionAndOpen()
                    1 -> pickImageFromGallery.launch("image/*")
                }
            }
            .setNegativeButton("Batal", null)
            .show()
    }

    private fun checkCameraPermissionAndOpen() {
        val permissionStatus = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        if (permissionStatus == PackageManager.PERMISSION_GRANTED) {
            openCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun openCamera() {
        val uri = createCameraImageUri()
        if (uri == null) {
            Toast.makeText(this, "Tidak dapat membuka kamera.", Toast.LENGTH_SHORT).show()
            return
        }
        cameraImageUri = uri
        takePictureFromCamera.launch(uri)
    }

    private fun createCameraImageUri(): Uri? {
        return try {
            val imagesDir = File(cacheDir, "images").also { it.mkdirs() }
            val imageFile = File(imagesDir, "captured_${System.currentTimeMillis()}.jpg")
            FileProvider.getUriForFile(this, "${packageName}.fileprovider", imageFile)
        } catch (e: Exception) {
            null
        }
    }

    // ─── Proses Gambar ────────────────────────────────────────────────────────

    private fun processSelectedImage(uri: Uri) {
        cardGemini.visibility = View.GONE
        tvResponsGemini.text = ""
        btnTanyaGemini.visibility = View.GONE
        setUiLoading(true)

        lifecycleScope.launch {
            try {
                // 1. Decode + perbaiki orientasi EXIF agar gambar tidak miring
                val bitmap = withContext(Dispatchers.IO) { uriToBitmapWithExif(uri) }

                ivPreview.setImageBitmap(bitmap)
                layoutPlaceholder.visibility = View.GONE
                ivPreview.visibility = View.VISIBLE

                // 2. Crop area wajah terlebih dahulu — model FANE ditraining dengan foto wajah yang sudah di-crop
                val faceBitmap = cropFaceFromBitmap(bitmap)

                // 3. Jalankan inference TFLite pada area wajah yang sudah di-crop
                val hasil: EmotionResult? = withContext(Dispatchers.Default) {
                    emotionDetector?.detectEmotion(faceBitmap)
                }

                when {
                    hasil != null -> tampilkanHasilDeteksi(hasil)
                    emotionDetector == null -> showModelError(modelInitError)
                    else -> showModelError("detectEmotion mengembalikan null")
                }
            } catch (e: Exception) {
                showModelError("Deteksi gagal: ${e.localizedMessage}")
            } finally {
                setUiLoading(false)
            }
        }
    }

    // ─── EXIF Rotation Fix ────────────────────────────────────────────────────

    private fun uriToBitmapWithExif(uri: Uri): Bitmap {
        val bitmap = decodeBitmapFromUri(uri)
        return applyExifRotation(bitmap, uri)
    }

    private fun decodeBitmapFromUri(uri: Uri): Bitmap {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.isMutableRequired = true
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
    }

    private fun applyExifRotation(bitmap: Bitmap, uri: Uri): Bitmap {
        return try {
            val inputStream = contentResolver.openInputStream(uri) ?: return bitmap
            val exif = ExifInterface(inputStream)
            inputStream.close()
            val degrees = when (
                exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            ) {
                ExifInterface.ORIENTATION_ROTATE_90  -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
            if (degrees == 0f) return bitmap
            val matrix = Matrix().apply { postRotate(degrees) }
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (rotated !== bitmap) bitmap.recycle()
            rotated
        } catch (e: Exception) {
            bitmap
        }
    }

    // ─── ML Kit Face Detection + Crop ─────────────────────────────────────────

    private suspend fun cropFaceFromBitmap(bitmap: Bitmap): Bitmap {
        return suspendCancellableCoroutine { cont ->
            val options = FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setMinFaceSize(0.1f)
                .build()

            val inputImage = InputImage.fromBitmap(bitmap, 0)

            FaceDetection.getClient(options).process(inputImage)
                .addOnSuccessListener { faces ->
                    if (faces.isNotEmpty()) {
                        // Ambil wajah terbesar (paling dekat ke kamera)
                        val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                        val cropped = safeCropWithPadding(bitmap, face.boundingBox, paddingFraction = 0.20f)
                        cont.resume(cropped)
                    } else {
                        // Tidak ada wajah terdeteksi — fallback ke center-crop 75%
                        cont.resume(centerCrop(bitmap, 0.75f))
                    }
                }
                .addOnFailureListener {
                    cont.resume(bitmap)
                }
        }
    }

    private fun safeCropWithPadding(bitmap: Bitmap, bounds: Rect, paddingFraction: Float): Bitmap {
        val size = maxOf(bounds.width(), bounds.height())
        val pad = (size * paddingFraction).toInt()
        val left  = maxOf(0, bounds.left - pad)
        val top   = maxOf(0, bounds.top - pad)
        val right  = minOf(bitmap.width,  bounds.right + pad)
        val bottom = minOf(bitmap.height, bounds.bottom + pad)
        return if (right > left && bottom > top) {
            Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
        } else {
            bitmap
        }
    }

    private fun centerCrop(bitmap: Bitmap, fraction: Float): Bitmap {
        val cropW = (bitmap.width * fraction).toInt()
        val cropH = (bitmap.height * fraction).toInt()
        val left = (bitmap.width - cropW) / 2
        val top  = (bitmap.height - cropH) / 2
        return Bitmap.createBitmap(bitmap, left, top, cropW, cropH)
    }

    // ─── Tampilkan Hasil Deteksi ──────────────────────────────────────────────

    private fun tampilkanHasilDeteksi(hasil: EmotionResult) {
        currentEmotion = hasil.label
        val confidence = hasil.confidence.coerceIn(0f, 100f)

        tvEmotionEmoji.text = getEmotionEmoji(currentEmotion)
        tvHasilEmosi.text = currentEmotion

        // Tampilkan raw scores + corrected scores untuk diagnosis
        val detector = emotionDetector
        val scoreText = if (detector != null) {
            val corrected = detector.labels.mapIndexed { i, lbl ->
                "$lbl ${String.format("%.0f", hasil.allScores[i] * 100)}%"
            }.joinToString(" | ")
            val raw = detector.labels.mapIndexed { i, lbl ->
                "$lbl ${String.format("%.0f", hasil.rawScores[i] * 100)}%"
            }.joinToString("|")
            "Hasil: $corrected\nRaw: $raw"
        } else {
            "Keyakinan: ${String.format("%.1f", confidence)}%"
        }
        tvConfidenceLabel.text = scoreText
        progressConfidence.progress = confidence.toInt()

        updateEmotionColor(currentEmotion)
        btnTanyaGemini.visibility = View.VISIBLE
    }

    private fun updateEmotionColor(emotion: String) {
        val colorRes = when (emotion.uppercase()) {
            "ANGRY"   -> R.color.color_angry
            "HAPPY"   -> R.color.color_happy
            "SAD"     -> R.color.color_sad
            "NEUTRAL" -> R.color.color_neutral
            else      -> R.color.primary
        }
        val color = ContextCompat.getColor(this, colorRes)
        cardResult.setCardBackgroundColor(ColorStateList.valueOf(color))
        progressConfidence.setIndicatorColor(color)
        btnTanyaGemini.setTextColor(ColorStateList.valueOf(color))
        btnTanyaGemini.setStrokeColorResource(colorRes)
    }

    private fun getEmotionEmoji(emotion: String): String {
        return when (emotion.uppercase()) {
            "ANGRY"   -> "😠"
            "HAPPY"   -> "😊"
            "SAD"     -> "😢"
            "NEUTRAL" -> "😐"
            else      -> "🤔"
        }
    }

    private fun setUiLoading(isLoading: Boolean) {
        btnPilihFoto.isEnabled = !isLoading
        progressBar.visibility = if (isLoading) View.VISIBLE else View.GONE
    }

    // ─── Gemini API ───────────────────────────────────────────────────────────

    private fun getGeminiApiKey(): String =
        getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
            .getString("gemini_api_key", "") ?: ""

    private fun saveGeminiApiKey(key: String) {
        getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
            .edit()
            .putString("gemini_api_key", key)
            .apply()
    }

    private fun mintaNasihatGemini(emosi: String) {
        val apiKey = getGeminiApiKey()
        if (apiKey.isBlank()) {
            showApiKeyDialog(emosi)
            return
        }
        jalankanGeminiRequest(emosi, apiKey)
    }

    private fun showApiKeyDialog(emosi: String) {
        val editText = EditText(this).apply {
            hint = "Masukkan Gemini API Key"
            setPadding(64, 40, 64, 40)
            setSingleLine(true)
        }
        AlertDialog.Builder(this)
            .setTitle("🔑 Gemini API Key")
            .setMessage("Dapatkan key GRATIS di:\naistudio.google.com\n\nKey disimpan di perangkat ini saja.")
            .setView(editText)
            .setPositiveButton("Simpan & Mulai") { _, _ ->
                val key = editText.text.toString().trim()
                if (key.isNotBlank()) {
                    saveGeminiApiKey(key)
                    jalankanGeminiRequest(emosi, key)
                } else {
                    Toast.makeText(this, "API Key tidak boleh kosong!", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Batal", null)
            .show()
    }

    private fun jalankanGeminiRequest(emosi: String, apiKey: String) {
        btnTanyaGemini.visibility = View.GONE
        progressBar.visibility = View.VISIBLE
        cardGemini.visibility = View.VISIBLE
        tvResponsGemini.text = "Gemini sedang mengetik..."

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val generativeModel = GenerativeModel(
                    modelName = "gemini-1.5-flash",
                    apiKey = apiKey
                )
                val promptText = """
                    Bertindaklah sebagai psikolog yang sangat empatik dan hangat.
                    User ini baru saja difoto dan AI mendeteksi emosinya sedang $emosi.
                    Berikan pesan yang sangat personal, menenangkan, dan berikan 2 saran
                    praktis yang spesifik untuk emosi tersebut agar hari mereka lebih baik.
                    Gunakan bahasa Indonesia yang gaul namun tetap sopan dan sangat
                    mendukung (supportive). Jawaban maksimal 150 kata.
                """.trimIndent()

                val response = generativeModel.generateContent(promptText)

                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    tvResponsGemini.text = response.text ?: "Tidak ada respons dari AI."
                    btnTanyaGemini.visibility = View.VISIBLE
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    progressBar.visibility = View.GONE
                    btnTanyaGemini.visibility = View.VISIBLE
                    val errMsg = e.localizedMessage ?: "Unknown error"
                    if (errMsg.contains("API_KEY_INVALID", ignoreCase = true) ||
                        errMsg.contains("401") || errMsg.contains("403")
                    ) {
                        saveGeminiApiKey("")
                        tvResponsGemini.text =
                            "❌ API Key tidak valid atau kadaluarsa.\n\nKetuk 'MINTA NASIHAT AI' lagi untuk memasukkan key baru."
                    } else {
                        tvResponsGemini.text =
                            "⚠️ Gagal menghubungi AI:\n$errMsg\n\nPastikan koneksi internet aktif."
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        emotionDetector?.close()
    }
}
