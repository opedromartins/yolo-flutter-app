// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.metadata.schema.ModelMetadata
import org.yaml.snakeyaml.Yaml
import java.nio.ByteBuffer
import java.nio.ByteOrder
import android.content.res.AssetManager

import org.json.JSONObject

import java.io.ByteArrayInputStream
import java.nio.MappedByteBuffer
import java.nio.charset.StandardCharsets

import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
/**
 * High-performance ObjectDetector that assumes no 90-degree rotation is needed
 * - Performs "resize -> getPixels -> ByteBuffer" in one pass, minimizing Canvas drawing
 * - Reuses Bitmap / ByteBuffer to reduce allocations
 * - Reuses inference output arrays
 */
class ObjectDetector(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private var numItemsThreshold: Int = 30,
    private val customOptions: Interpreter.Options? = null
) : BasePredictor() {
    // Inference output dimensions
    private var out1 = 0
    private var out2 = 0
    // Three image processors: camera portrait, camera landscape, and single images
    private lateinit var imageProcessorCameraPortrait: ImageProcessor
    private lateinit var imageProcessorCameraPortraitFront: ImageProcessor
    private lateinit var imageProcessorCameraLandscape: ImageProcessor
    private lateinit var imageProcessorSingleImage: ImageProcessor


//    companion object {
//
//    }
    // Reuse inference output array ([1][out1][out2])
    private lateinit var rawOutput: Array<Array<FloatArray>>
    // Transposed array for post-processing
    private lateinit var predictions: Array<FloatArray>

    // ======== Workspace for fast preprocessing ========
    // (1) Temporary scaled Bitmap matching model input size
    //     No 90-degree rotation needed, so simply cache createScaledBitmap() equivalent
    private lateinit var scaledBitmap: Bitmap

    // (2) Array to temporarily store pixels (inWidth*inHeight)
    private lateinit var intValues: IntArray

    // (3) ByteBuffer for TFLite input (1 * height * width * 3 * 4 bytes)
    private lateinit var inputBuffer: ByteBuffer

    // Options for TensorFlow Lite Interpreter
    private val interpreterOptions: Interpreter.Options = (customOptions ?: Interpreter.Options()).apply {
        // If no custom options provided, use default threads
        if (customOptions == null) {
            setNumThreads(Runtime.getRuntime().availableProcessors())
        }
        
        // If customOptions is provided, only add GPU delegate if requested
        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
                Log.d("ObjectDetector", "GPU delegate is used.")
            } catch (e: Exception) {
                Log.e("ObjectDetector", "GPU delegate error: ${e.message}")
            }
        }
    }

    // ========== TFLite Interpreter ==========
    // Use protected var interpreter: Interpreter? = null from BasePredictor if available
    // Otherwise, keep it in this class as usual
    init {
        val assetManager = context.assets
        val modelBuffer  = YOLOUtils.loadModelFile(context, modelPath)

        /* --- Get labels from metadata (try Appended ZIP → FlatBuffers in order) --- */
        var loadedLabels = YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)
        var labelsWereLoaded = loadedLabels != null

        if (labelsWereLoaded) {
            this.labels = loadedLabels!! // Use labels from appended ZIP
            Log.i(TAG, "Labels successfully loaded from appended ZIP.")
        } else {
            Log.w(TAG, "Could not load labels from appended ZIP, trying FlatBuffers metadata...")
            // Try FlatBuffers as a fallback
            if (loadLabelsFromFlatbuffers(modelBuffer)) {
                labelsWereLoaded = true
                Log.i(TAG, "Labels successfully loaded from FlatBuffers metadata.")
            }
        }

        if (!labelsWereLoaded) {
            Log.w(TAG, "No embedded labels found from appended ZIP or FlatBuffers. Using labels passed via constructor (if any) or an empty list.")
            // If labels were passed via constructor and not overridden, they will be used.
            // If no labels were passed and none loaded, this.labels will be what was passed or an uninitialized/empty list
            // depending on how the 'labels' property was handled if it was nullable or had a default.
            // Given 'override var labels: List<String>' is passed in constructor, it will hold the passed value.
            if (this.labels.isEmpty()) {
                 Log.w(TAG, "Warning: No labels loaded and no labels provided via constructor. Detections might lack class names.")
            }
        }

        interpreter = Interpreter(modelBuffer, interpreterOptions)
        // Call allocateTensors() once during initialization, not in the inference loop
        interpreter.allocateTensors()
        Log.d("TAG", "TFLite model loaded: $modelPath, tensors allocated")

        // Check input shape (example: [1, inHeight, inWidth, 3])
        val inputShape = interpreter.getInputTensor(0).shape()
        val inBatch = inputShape[0]         // Usually 1
        val inHeight = inputShape[1]        // Example: 320
        val inWidth = inputShape[2]         // Example: 320
        val inChannels = inputShape[3]      // 3 (RGB)
        require(inBatch == 1 && inChannels == 3) {
            "Input tensor shape not supported. Expected [1, H, W, 3]. But got ${inputShape.joinToString()}"
        }
        inputSize = Size(inWidth, inHeight) // Set variable in BasePredictor
        modelInputSize = Pair(inWidth, inHeight)
        Log.d("TAG", "Model input size = $inWidth x $inHeight")

        // Output shape (varies by model, modify as needed)
        // Example: [1, 84, 2100] = [batch, outHeight, outWidth]
        val outputShape = interpreter.getOutputTensor(0).shape()
        out1 = outputShape[1] // 84
        out2 = outputShape[2] // 2100
        Log.d("TAG", "Model output shape = [1, $out1, $out2]")

        // Allocate preprocessing resources
        initPreprocessingResources(inWidth, inHeight)

        // Allocate inference output arrays
        rawOutput = Array(1) { Array(out1) { FloatArray(out2) } }
        predictions = Array(out2) { FloatArray(out1) }
        
        // Initialize three image processors:
        
        // 1. For camera feed in portrait mode - includes 270-degree rotation
        imageProcessorCameraPortrait = ImageProcessor.Builder()
            .add(Rot90Op(3))  // 270-degree rotation (3 * 90 degrees) for back camera
            .add(ResizeOp(inputSize.height, inputSize.width, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
            .add(CastOp(INPUT_IMAGE_TYPE))
            .build()
            
        // 2. For front camera in portrait mode - 90-degree rotation
        imageProcessorCameraPortraitFront = ImageProcessor.Builder()
            .add(Rot90Op(1))  // 90-degree rotation for front camera
            .add(ResizeOp(inputSize.height, inputSize.width, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
            .add(CastOp(INPUT_IMAGE_TYPE))
            .build()
            
        // 3. For camera feed in landscape mode - no rotation needed
        imageProcessorCameraLandscape = ImageProcessor.Builder()
            .add(ResizeOp(inputSize.height, inputSize.width, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
            .add(CastOp(INPUT_IMAGE_TYPE))
            .build()
            
        // 4. For single images - no rotation needed
        imageProcessorSingleImage = ImageProcessor.Builder()
            .add(ResizeOp(inputSize.height, inputSize.width, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
            .add(CastOp(INPUT_IMAGE_TYPE))
            .build()
            
        Log.d("TAG", "ObjectDetector initialized.")
    }

    /* =================================================================== */
    /*                 metadata helper functions (Kotlin)                 */
    /* =================================================================== */

    // Old ZIP loading methods (readWholeModel, findPKHeader, loadLabelsFromEmbeddedZip)
    // have been removed as their functionality is replaced by YOLOFileUtils.loadLabelsFromAppendedZip

    /**
     * ────────────────────────────────────────────────────────────────
     *  Load labels from FlatBuffers (metadata.yaml) - based on old code
     *  - Scan all associatedFileNames
     *  - Parse YAML as Map<Int,String>
     *  - Use values directly as List and assign to labels
     * ────────────────────────────────────────────────────────────────
     */
    private fun loadLabelsFromFlatbuffers(buf: MappedByteBuffer): Boolean = try {
        val extractor = MetadataExtractor(buf)
        val files = extractor.associatedFileNames
        if (!files.isNullOrEmpty()) {
            for (fileName in files) {
                Log.d(TAG, "Found associated file: $fileName")
                extractor.getAssociatedFile(fileName)?.use { stream ->
                    val fileString = String(stream.readBytes(), Charsets.UTF_8)
                    Log.d(TAG, "Associated file contents:\n$fileString")

                    val yaml = Yaml()
                    @Suppress("UNCHECKED_CAST")
                    val data = yaml.load<Map<String, Any>>(fileString)
                    if (data != null && data.containsKey("names")) {
                        val namesMap = data["names"] as? Map<Int, String>
                        if (namesMap != null) {
                            labels = namesMap.values.toList()          // Same as old code
                            Log.d(TAG, "Loaded labels from metadata: $labels")
                            return true
                        }
                    }
                }
            }
        } else {
            Log.d(TAG, "No associated files found in the metadata.")
        }
        false
    } catch (e: Exception) {
        Log.e(TAG, "Failed to extract metadata: ${e.message}")
        false
    }


    private fun initPreprocessingResources(width: Int, height: Int) {
        // ARGB_8888 Bitmap for input size (e.g., 320x320)
        scaledBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Int array for pixel reading
        intValues = IntArray(width * height)

        // Buffer for TFLite input
        inputBuffer = ByteBuffer.allocateDirect(1 * width * height * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
    }

    /**
     * Main inference method
     * @param bitmap Input bitmap to process
     * @param origWidth Original width of the source image
     * @param origHeight Original height of the source image
     * @param rotateForCamera Whether this is a camera feed that requires rotation (true) or a single image (false)
     * @param isLandscape Whether the device is in landscape orientation
     */
    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean, isLandscape: Boolean): YOLOResult {
        val overallStartTime = System.nanoTime()
        var stageStartTime = overallStartTime

        inputBuffer.clear()

        val processedImage = if (rotateForCamera) {
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val proc = if (isLandscape) imageProcessorCameraLandscape
                else if (isFrontCamera) imageProcessorCameraPortraitFront
                else imageProcessorCameraPortrait
            proc.process(tensorImage)
        } else {
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            imageProcessorSingleImage.process(tensorImage)
        }

        inputBuffer.put(processedImage.buffer)
        inputBuffer.rewind()

        val preprocessTimeMs = (System.nanoTime() - stageStartTime) / 1_000_000.0
        Log.d(TAG, "Predict Stage: Preprocessing done in $preprocessTimeMs ms")
        stageStartTime = System.nanoTime()

        Log.d(TAG, "Predict Start: Inference")
        interpreter.run(inputBuffer, rawOutput)
        val inferenceTimeMs = (System.nanoTime() - stageStartTime) / 1_000_000.0
        Log.d(TAG, "Predict Stage: Inference done in $inferenceTimeMs ms")
        stageStartTime = System.nanoTime()

        Log.d(TAG, "Predict Start: Postprocessing")
        val outHeight = rawOutput[0].size
        val outWidth = rawOutput[0][0].size
        val shape = interpreter.getOutputTensor(0).shape()
        Log.d(TAG, "Output shape: ${shape.contentToString()}")

        val mw = inputSize.width.toFloat()
        val mh = inputSize.height.toFloat()

        // YOLO26 e2e: [1, 300, 6] = [x1,y1,x2,y2,conf,class] em pixels. JNI causa crash!
        val isE2E = outWidth == 6 && outHeight in 200..500

        val resultBoxes = if (isE2E) {
            postprocessE2E(rawOutput[0], confidenceThreshold, numItemsThreshold)
        } else {
            for (i in 0 until outHeight) {
                for (j in 0 until outWidth) {
                    predictions[j][i] = rawOutput[0][i][j]
                }
            }
            postprocess(predictions, outHeight, outWidth, confidenceThreshold, iouThreshold, numItemsThreshold, labels.size)
        }

        val boxes = mutableListOf<Box>()
        for (boxArray in resultBoxes) {
            if (boxArray.size < 6) continue
            val conf = boxArray[4]
            val classIdx = boxArray[5].toInt()

            val left: Float
            val top: Float
            val right: Float
            val bottom: Float
            if (isE2E) {
                val x1 = boxArray[0]; val y1 = boxArray[1]
                val x2 = boxArray[2]; val y2 = boxArray[3]
                // Ultralytics e2e: coordenadas em espaço do input (pixels) ou 0-1
                val scaleX = if (x2 <= 1f && y2 <= 1f) origWidth.toFloat() else origWidth / mw
                val scaleY = if (x2 <= 1f && y2 <= 1f) origHeight.toFloat() else origHeight / mh
                left = (x1 * scaleX).coerceIn(0f, origWidth.toFloat())
                top = (y1 * scaleY).coerceIn(0f, origHeight.toFloat())
                right = (x2 * scaleX).coerceIn(0f, origWidth.toFloat())
                bottom = (y2 * scaleY).coerceIn(0f, origHeight.toFloat())
            } else {
                val x1 = boxArray[0]; val y1 = boxArray[1]
                val x2 = boxArray[2]; val y2 = boxArray[3]
                left = (x1 * origWidth).coerceIn(0f, origWidth.toFloat())
                top = (y1 * origHeight).coerceIn(0f, origHeight.toFloat())
                right = (x2 * origWidth).coerceIn(0f, origWidth.toFloat())
                bottom = (y2 * origHeight).coerceIn(0f, origHeight.toFloat())
            }

            val rect = RectF(left, top, right, bottom)
            val normRect = RectF(left / origWidth, top / origHeight, right / origWidth, bottom / origHeight)

            if (rect.width() > 1f && rect.height() > 1f) {
                val label = if (classIdx in labels.indices) labels[classIdx] else "class_$classIdx"
                boxes.add(Box(classIdx, label, conf, rect, normRect))
            }
        }

        val postprocessTimeMs = (System.nanoTime() - stageStartTime) / 1_000_000.0
        val totalMs = (System.nanoTime() - overallStartTime) / 1_000_000.0
        Log.d(TAG, "Predict Total time: $totalMs ms (Pre: $preprocessTimeMs, Inf: $inferenceTimeMs, Post: $postprocessTimeMs)")

        updateTiming()
        return YOLOResult(
            origShape = com.ultralytics.yolo.Size(origWidth, origHeight),
            boxes = boxes,
            speed = totalMs,
            fps = if (t4 > 0.0) 1.0 / t4 else 0.0,
            names = labels
        )
    }
    // Thresholds (like setConfidenceThreshold, setIouThreshold in TFLiteDetector)
    private var confidenceThreshold = 0.25f
    private var iouThreshold = 0.4f
//    private var numItemsThreshold = 30

    override fun setConfidenceThreshold(conf: Double) {
        confidenceThreshold = conf.toFloat()
        super.setConfidenceThreshold(conf)
    }

    override fun setIouThreshold(iou: Double) {
        iouThreshold = iou.toFloat()
        super.setIouThreshold(iou)
    }

    override fun getConfidenceThreshold(): Double {
        return confidenceThreshold.toDouble()
    }

    override fun getIouThreshold(): Double {
        return iouThreshold.toDouble()
    }

    override fun setNumItemsThreshold(n: Int) {
        numItemsThreshold = n
        super.setNumItemsThreshold(n)
    }

    /** E2E format [N,6]: x1,y1,x2,y2,conf,class. Coordenadas podem ser 0-1 (norm) ou pixels. Ordena por confiança. */
    private fun postprocessE2E(raw: Array<FloatArray>, confThr: Float, maxDet: Int): Array<FloatArray> {
        val out = mutableListOf<FloatArray>()
        for (row in raw) {
            if (row.size < 6) continue
            if (row[4] < confThr) continue
            out.add(floatArrayOf(row[0], row[1], row[2], row[3], row[4], row[5]))
        }
        return out.sortedByDescending { it[4] }.take(maxDet).toTypedArray()
    }

    // Post-processing via JNI
    private external fun postprocess(
        predictions: Array<FloatArray>,
        w: Int,
        h: Int,
        confidenceThreshold: Float,
        iouThreshold: Float,
        numItemsThreshold: Int,
        numClasses: Int
    ): Array<FloatArray>

    companion object {
        private const val TAG = "ObjectDetector"
        // Load JNI library
        init {
            System.loadLibrary("ultralytics")
        }
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.25F
        private const val IOU_THRESHOLD = 0.4F
    }
}