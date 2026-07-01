package com.example.aiemotiondetector

// appV1.0 Rev 6 (MainActivity.kt)
// Fix: Error model AI kini tampil di card UI (bukan hanya Toast)
// Fix: EmotionDetector tidak lagi bergantung tensorflow-lite-support
// Fix: Tombol kini retry init model otomatis (model sebelumnya gagal load
// karena FULLY_CONNECTED v12 tidak didukung tensorflow-lite:2.16.1 — sudah
// diganti ke LiteRT 1.0.1 yang mendukung op tersebut)
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.ImageDecoder
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    // Nullable: aman di onDestroy meski init model gagal
    private var emotionDetector: EmotionDetector? = null
    private var modelInitError: String = ""
    private var currentEmotion: String = ""

    // URI sementara untuk menyimpan hasil foto kamera
    private var cameraImageUri: Uri? = null

    // UI Components
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

    // ─── Activity Result: Ambil dari Galeri ──────────────────────────────────
    private val pickImageFromGallery =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { processSelectedImage(it) }
        }

    // ─── Activity Result: Ambil Foto dari Kamera ─────────────────────────────
    private val takePictureFromCamera =
        registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
            if (success) {
                cameraImageUri?.let { processSelectedImage(it) }
            } else {
                Toast.makeText(this, "Pengambilan foto dibatalkan.", Toast.LENGTH_SHORT).show()
            }
        }

    // ─── Activity Result: Minta Izin Kamera ──────────────────────────────────
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

        // Binding semua UI component
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

        // Inisialisasi TFLite model — error disimpan untuk ditampilkan di UI
        initModel()

        btnPilihFoto.setOnClickListener {
            if (emotionDetector == null) {
                // Coba init ulang sebelum menyerah — berguna setelah fix dependency
                // tanpa user harus force-stop aplikasi secara manual
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

    // Coba inisialisasi TFLite model. Dipanggil di onCreate, dan dicoba ulang
    // saat tombol diklik jika percobaan pertama gagal.
    private fun initModel() {
        if (emotionDetector != null) return
        try {
            emotionDetector = EmotionDetector(this)
        } catch (e: Exception) {
            modelInitError = e.localizedMessage ?: "Unknown error saat load model"
            showModelError(modelInitError)
        }
    }

    // Tampilkan error model di card result agar tidak terlewat seperti Toast
    private fun showModelError(errorMsg: String) {
        val errorColor = ContextCompat.getColor(this, R.color.error)
        cardResult.setCardBackgroundColor(ColorStateList.valueOf(errorColor))
        tvEmotionEmoji.text = "⚠️"
        tvHasilEmosi.text = "Model Gagal Load"
        tvConfidenceLabel.text = "Error: $errorMsg"
        progressConfidence.progress = 0
    }

    // ─── Dialog Pilih Sumber Foto ─────────────────────────────────────────────

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

    // ─── Kamera: Cek Izin → Buka ─────────────────────────────────────────────

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

    // Buat URI sementara di cache untuk foto kamera (wajib Android 7+ via FileProvider)
    private fun createCameraImageUri(): Uri? {
        return try {
            val imagesDir = File(cacheDir, "images").also { it.mkdirs() }
            val imageFile = File(imagesDir, "captured_${System.currentTimeMillis()}.jpg")
            FileProvider.getUriForFile(this, "${packageName}.fileprovider", imageFile)
        } catch (e: Exception) {
            null
        }
    }

    // ─── Proses Gambar (dipanggil dari kamera MAUPUN galeri) ──────────────────

    private fun processSelectedImage(uri: Uri) {
        // Reset state sebelum proses gambar baru
        cardGemini.visibility = View.GONE
        tvResponsGemini.text = ""
        btnTanyaGemini.visibility = View.GONE
        setUiLoading(true)

        lifecycleScope.launch {
            try {
                // Decode gambar di IO thread agar UI tidak freeze
                val bitmap = withContext(Dispatchers.IO) { uriToBitmap(uri) }

                ivPreview.setImageBitmap(bitmap)
                layoutPlaceholder.visibility = View.GONE
                ivPreview.visibility = View.VISIBLE

                // Jalankan TFLite inference di Default thread
                val hasil = withContext(Dispatchers.Default) {
                    emotionDetector?.detectEmotion(bitmap)
                }

                when {
                    hasil != null -> tampilkanHasilDeteksi(hasil)
                    emotionDetector == null -> showModelError(modelInitError)
                    else -> showModelError("detectEmotion mengembalikan null")
                }
            } catch (e: Exception) {
                // Tampilkan error inference di card agar tidak terlewat
                showModelError("Deteksi gagal: ${e.localizedMessage}")
            } finally {
                setUiLoading(false)
            }
        }
    }

    // ─── Tampilkan Hasil Deteksi Emosi ───────────────────────────────────────

    private fun tampilkanHasilDeteksi(hasil: Pair<String, Float>) {
        currentEmotion = hasil.first
        val confidence = hasil.second.coerceIn(0f, 100f)

        tvEmotionEmoji.text = getEmotionEmoji(currentEmotion)
        tvHasilEmosi.text = currentEmotion
        tvConfidenceLabel.text = "Keyakinan: ${String.format("%.1f", confidence)}%"
        progressConfidence.progress = confidence.toInt()

        updateEmotionColor(currentEmotion)
        btnTanyaGemini.visibility = View.VISIBLE
    }

    // Warna card berubah dinamis sesuai emosi yang terdeteksi
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

    // ─── Gemini API Logic ─────────────────────────────────────────────────────

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

    // Dialog input API key — cukup sekali, disimpan di SharedPreferences
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
                    // Jika key tidak valid, hapus agar user bisa input ulang
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

    // ─── Helper ──────────────────────────────────────────────────────────────

    private fun uriToBitmap(uri: Uri): Bitmap {
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

    override fun onDestroy() {
        super.onDestroy()
        emotionDetector?.close()
    }
}
