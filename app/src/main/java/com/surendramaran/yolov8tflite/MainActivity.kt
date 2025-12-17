package com.surendramaran.yolov8tflite

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding

    private var detector: Detector? = null
    private val PICK_IMAGE = 100
    private var selectedImageUri: Uri? = null

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Инициализация детектора в фоновом потоке
        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
        }

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Кнопка для выбора фото из галереи
        binding.buttonLoadImage.setOnClickListener {
            openGallery()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        cameraExecutor.shutdown()
    }

    // Проверка прав
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
    }

    // Открываем галерею
    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = "image/*"
        startActivityForResult(intent, PICK_IMAGE)
    }

    // Обрабатываем выбор изображения
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_IMAGE && resultCode == Activity.RESULT_OK) {
            data?.data?.let { uri ->
                selectedImageUri = uri
                processGalleryImage(uri)
            }
        }
    }

    // Отображаем Bitmap в ImageView
    private fun showImage(bitmap: Bitmap) {
        runOnUiThread {
            binding.imageView.setImageBitmap(bitmap)
            binding.inferenceTime.text = ""
        }
    }

    // Обработка загруженного фото: ресайз, показ, запуск детекции в фоне с измерением времени
    private fun processGalleryImage(uri: Uri) {
        try {
            val inputStream = contentResolver.openInputStream(uri)
            var bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            // Ресайз под размер модели (пример 320x320)
            bitmap = Bitmap.createScaledBitmap(bitmap, 320, 320, true)

            showImage(bitmap)

            cameraExecutor.submit {
                val startTime = System.nanoTime()
                detector?.detect(bitmap)
                val endTime = System.nanoTime()
                val durationMs = (endTime - startTime) / 1_000_000

                runOnUiThread {
                    binding.inferenceTime.text = "Inference time: $durationMs ms"
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing gallery image", e)
        }
    }

    // Получение результатов детекции
    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime} ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }
        }
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.clear()
        }
    }
}
