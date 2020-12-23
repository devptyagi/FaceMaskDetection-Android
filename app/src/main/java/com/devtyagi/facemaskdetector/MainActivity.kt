package com.devtyagi.facemaskdetector

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.Image
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Log
import android.widget.Toast
import androidx.appcompat.content.res.AppCompatResources
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.devtyagi.facemaskdetector.ml.FackMaskDetection
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.model.Model
import java.lang.Exception
import java.lang.IllegalStateException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

typealias CameraBitmapOutputListener = (bitmap: Bitmap) -> Unit

class MainActivity : AppCompatActivity() {

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing: Int = CameraSelector.LENS_FACING_FRONT
    private var camera: Camera? = null
    private lateinit var cameraExecuter: ExecutorService
    private lateinit var faceMaskDetection: FackMaskDetection

    companion object{
        private const val TAG = "Face Mask Detector"
        private const val REQUEST_CODE_PERMISSIONS = 0x98
        private val REQUIRED_PERMISSIONS: Array<String> = arrayOf(Manifest.permission.CAMERA)
        private const val RATIO_4_3_VALUE: Double = (4.0 / 3.0)
        private const val RATIO_16_9_VALUE: Double = (16.0 / 9.0)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        if(!allPermissionsGranted) {
            requestCameraPermissions()
        } else {
            setupML()
            setupCameraThread()
            setupCameraControllers()
            setupCamera()
        }
    }

    private fun setupML() {
        val options: Model.Options = Model.Options.Builder().setDevice(Model.Device.GPU).setNumThreads(5).build()
        faceMaskDetection = FackMaskDetection.newInstance(applicationContext, options)
    }

    private fun requestCameraPermissions() {
        ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
    }

    private fun setupCameraControllers() {
        fun setLensButtonIcon() {
            btnCameraLensFace.setImageDrawable(
                    AppCompatResources.getDrawable(
                            applicationContext,
                            if(lensFacing == CameraSelector.LENS_FACING_FRONT) {
                                R.drawable.ic_baseline_camera_rear_24
                            } else {
                                R.drawable.ic_baseline_camera_front_24
                            }
                    )
            )
        }
        setLensButtonIcon()
        btnCameraLensFace.setOnClickListener{
            lensFacing = if(lensFacing == CameraSelector.LENS_FACING_FRONT) {
                CameraSelector.LENS_FACING_BACK
            } else {
                CameraSelector.LENS_FACING_FRONT
            }
            setLensButtonIcon()
            setupCameraUseCases()
        }
        try {
            btnCameraLensFace.isEnabled = hasBackCamera && hasFrontCamera
        } catch (exception: CameraInfoUnavailableException) {
            btnCameraLensFace.isEnabled = false
        }
    }

    private fun setupCameraUseCases() {
        val cameraSelector: CameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        val metrics: DisplayMetrics = DisplayMetrics().also { previewView.display.getRealMetrics(it) }
        val rotation: Int = previewView.display.rotation
        val screenAspectRatio: Int = aspectRatio(metrics.widthPixels, metrics.heightPixels)
        preview = Preview.Builder()
                .setTargetAspectRatio(screenAspectRatio)
                .setTargetRotation(rotation)
                .build()
        imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(screenAspectRatio)
                .setTargetRotation(rotation)
                .build()
                .also { it.setAnalyzer(
                        cameraExecuter, BitmapOutputAnalysis(applicationContext) {
                    bitmap -> setupMLOutput(bitmap)
                }
                ) }
        cameraProvider?.unbindAll()
        try {
            camera = cameraProvider?.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
            )
            preview?.setSurfaceProvider(previewView.surfaceProvider)
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failure", e)
        }
    }

    private fun setupMLOutput(bitmap: Bitmap) {
        val tensorImage: TensorImage = TensorImage.fromBitmap(bitmap)
        val results: FackMaskDetection.Outputs = faceMaskDetection.process(tensorImage)
        val output: List<Category> = results.probabilityAsCategoryList.apply {
            sortByDescending { res -> res.score }
        }
        lifecycleScope.launch(Dispatchers.Main) {
            output.firstOrNull()?.let {
                category ->
                txtOutput.text = category.label
                txtOutput.setTextColor(
                        ContextCompat.getColor(
                                applicationContext,
                                if(category.label == "without_mask") R.color.red else R.color.green
                        )
                )
                overlay.background = getDrawable(if (category.label == "without_mask") R.drawable.red_border else R.drawable.green_border)
                progressOutput.progressTintList = AppCompatResources.getColorStateList(
                        applicationContext,
                        if(category.label == "without_mask") R.color.red else R.color.green
                )
                progressOutput.progress = (category.score * 100).toInt()
            }
        }
    }

    private fun setupCameraThread() {
        cameraExecuter = Executors.newSingleThreadExecutor()
    }

    private fun grantedCameraPermission(requestCode: Int) {
        if(requestCode == REQUEST_CODE_PERMISSIONS) {
            if(allPermissionsGranted) {
                setupCamera()
            } else {
                Toast.makeText(this, "Permission not granted", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    private fun setupCamera() {
        val cameraProviderFuture: ListenableFuture<ProcessCameraProvider> = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            cameraProvider = cameraProviderFuture.get()
            lensFacing = when{
                hasFrontCamera -> CameraSelector.LENS_FACING_FRONT
                hasBackCamera -> CameraSelector.LENS_FACING_BACK
                else -> throw IllegalStateException("No Camera Available")
            }
            setupCameraControllers()
            setupCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private val allPermissionsGranted: Boolean get() {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private val hasBackCamera: Boolean get() {
        return cameraProvider?.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) ?: false
    }

    private val hasFrontCamera: Boolean get() {
        return cameraProvider?.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA) ?: false
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        grantedCameraPermission(requestCode)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        setupCameraControllers()
    }

    private fun aspectRatio(width: Int, height: Int): Int {
        val previewRatio: Double = max(width, height).toDouble() / min(width, height)
        if(abs(previewRatio - RATIO_4_3_VALUE) <= abs(previewRatio - RATIO_16_9_VALUE)) {
            return AspectRatio.RATIO_4_3
        }
        return AspectRatio.RATIO_16_9
    }

}

private class BitmapOutputAnalysis(
        context: Context,
        private val listener: CameraBitmapOutputListener
): ImageAnalysis.Analyzer {
    private val yuvToRGBConverter = YuvToRGBConverter(context)
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var rotationMatrix: Matrix

    @SuppressLint("UnsafeExperimentalUsageError")
    private fun ImageProxy.toBitmap(): Bitmap? {
        val image: Image = this.image ?: return null
        if(!::bitmapBuffer.isInitialized) {
            rotationMatrix = Matrix()
            rotationMatrix.postRotate(this.imageInfo.rotationDegrees.toFloat())
            bitmapBuffer = Bitmap.createBitmap(this.width, this.height, Bitmap.Config.ARGB_8888)
        }
        yuvToRGBConverter.yuvToRgb(image, bitmapBuffer)
        return Bitmap.createBitmap(
                bitmapBuffer,
                0,
                0,
                bitmapBuffer.width,
                bitmapBuffer.height,
                rotationMatrix,
                false
        )
    }

    override fun analyze(image: ImageProxy) {
        image.toBitmap()?.let {
            listener(it)
        }
        image.close()
    }

}