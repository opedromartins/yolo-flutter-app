// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.yaml.snakeyaml.Yaml
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.charset.StandardCharsets
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Object detector with Ultralytics-aligned preprocessing: letterbox (stride-free resize + pad 114)
 * and the same default thresholds as COCO eval (low conf, NMS IoU 0.7, max detections 300).
 */
class ObjectDetector(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private var numItemsThreshold: Int = 300,
    private val customOptions: Interpreter.Options? = null
) : BasePredictor() {
    // Inference output dimensions
    private var out1 = 0
    private var out2 = 0

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
        // If no custom options provided, use default threads + XNNPACK (CPU mais rápido)
        if (customOptions == null) {
            setNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 8))
            try {
                setUseXNNPACK(true)
            } catch (e: Exception) {
                Log.w("ObjectDetector", "setUseXNNPACK not applied: ${e.message}")
            }
        }

        // If customOptions is provided, only add GPU delegate if requested
        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
                Log.d("ObjectDetector", "GPU delegate is used.")
            } catch (e: Exception) {
                Log.e("ObjectDetector", "GPU delegate error: ${e.message}")
            }
        } else {
            // GPU desligado: tenta NNAPI (NPU/DSP em muitos telemóveis) — pode falhar em alguns dispositivos
            try {
                addDelegate(NnApiDelegate())
                Log.d("ObjectDetector", "NNAPI delegate is used (useGpu=false).")
            } catch (e: Exception) {
                Log.e("ObjectDetector", "NNAPI delegate error: ${e.message}")
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

    /** Letterbox metadata matching Ultralytics val: resize + center pad with color 114. */
    private data class LetterboxParams(
        val gain: Float,
        val padX: Float,
        val padY: Float,
        val inW: Float,
        val inH: Float
    )

    /** Same as TensorFlow Rot90Op(k): k rotations, 90° counter-clockwise each. */
    private fun rotateK90CounterClockwise(src: Bitmap, k: Int): Bitmap {
        val kk = ((k % 4) + 4) % 4
        if (kk == 0) return src
        val m = Matrix()
        m.postRotate(-90f * kk)
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    /**
     * Draw letterboxed image on [scaledBitmap], fill [buffer] with RGB float32 /255 (NHWC).
     */
    private fun letterboxToInputBuffer(
        src: Bitmap,
        outW: Int,
        outH: Int,
        buffer: ByteBuffer
    ): LetterboxParams {
        val w = src.width.toFloat()
        val h = src.height.toFloat()
        val r = min(outW / w, outH / h)
        val newW = max(1, (w * r).roundToInt())
        val newH = max(1, (h * r).roundToInt())
        val padW = outW - newW
        val padH = outH - newH
        val padX = padW / 2f
        val padY = padH / 2f

        val canvas = Canvas(scaledBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)
        canvas.drawBitmap(resized, padX, padY, null)
        if (resized != src) {
            resized.recycle()
        }

        buffer.clear()
        scaledBitmap.getPixels(intValues, 0, outW, 0, 0, outW, outH)
        var idx = 0
        for (i in 0 until outH) {
            for (j in 0 until outW) {
                val pixel = intValues[idx++]
                buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
                buffer.putFloat((pixel and 0xFF) / 255.0f)
            }
        }
        buffer.rewind()
        return LetterboxParams(r, padX, padY, outW.toFloat(), outH.toFloat())
    }

    private fun mapLetterboxNormXywhToOriginal(
        x: Float,
        y: Float,
        bw: Float,
        bh: Float,
        origW: Int,
        origH: Int,
        lb: LetterboxParams
    ): RectF {
        val x1p = x * lb.inW
        val y1p = y * lb.inH
        val x2p = x1p + bw * lb.inW
        val y2p = y1p + bh * lb.inH
        val left = ((x1p - lb.padX) / lb.gain).coerceIn(0f, origW.toFloat())
        val top = ((y1p - lb.padY) / lb.gain).coerceIn(0f, origH.toFloat())
        val right = ((x2p - lb.padX) / lb.gain).coerceIn(0f, origW.toFloat())
        val bottom = ((y2p - lb.padY) / lb.gain).coerceIn(0f, origH.toFloat())
        return RectF(left, top, right, bottom)
    }

    private fun mapLetterboxE2EToOriginal(
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        origW: Int,
        origH: Int,
        lb: LetterboxParams
    ): RectF {
        val looksNormalized = x2 <= 1f && y2 <= 1f && x1 <= 1f && y1 <= 1f && x2 >= x1 && y2 >= y1
        return if (looksNormalized) {
            val x1p = x1 * lb.inW
            val y1p = y1 * lb.inH
            val x2p = x2 * lb.inW
            val y2p = y2 * lb.inH
            val left = ((x1p - lb.padX) / lb.gain).coerceIn(0f, origW.toFloat())
            val top = ((y1p - lb.padY) / lb.gain).coerceIn(0f, origH.toFloat())
            val right = ((x2p - lb.padX) / lb.gain).coerceIn(0f, origW.toFloat())
            val bottom = ((y2p - lb.padY) / lb.gain).coerceIn(0f, origH.toFloat())
            RectF(left, top, right, bottom)
        } else {
            val left = ((x1 - lb.padX) / lb.gain).coerceIn(0f, origW.toFloat())
            val top = ((y1 - lb.padY) / lb.gain).coerceIn(0f, origH.toFloat())
            val right = ((x2 - lb.padX) / lb.gain).coerceIn(0f, origW.toFloat())
            val bottom = ((y2 - lb.padY) / lb.gain).coerceIn(0f, origH.toFloat())
            RectF(left, top, right, bottom)
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

        var rotatedBitmap: Bitmap? = null
        val oriented: Bitmap = when {
            !rotateForCamera -> bitmap
            isLandscape -> bitmap
            else -> {
                val k = if (isFrontCamera) 1 else 3
                rotateK90CounterClockwise(bitmap, k).also { rotatedBitmap = it }
            }
        }

        val lb = letterboxToInputBuffer(
            oriented,
            inputSize.width,
            inputSize.height,
            inputBuffer
        )
        rotatedBitmap?.recycle()

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

        // YOLO26 e2e: [1, 300, 6] = [x1,y1,x2,y2,conf,class] em pixels. JNI causa crash!
        val isE2E = outWidth == 6 && outHeight in 200..500

        val resultBoxes = if (isE2E) {
            postprocessE2E(rawOutput[0], confidenceThreshold, numItemsThreshold)
        } else {
            // JNI postprocess expects [channels][num_preds] where:
            // - channels = 4 + num_classes
            // - num_preds = number of candidate boxes
            // rawOutput[0] is already in this layout for common YOLO TFLite exports.
            postprocess(rawOutput[0], outWidth, outHeight, confidenceThreshold, iouThreshold, numItemsThreshold, labels.size)
        }

        val boxes = mutableListOf<Box>()
        for (boxArray in resultBoxes) {
            if (boxArray.size < 6) continue
            val conf = boxArray[4]
            val classIdx = boxArray[5].toInt()

            val rect = if (isE2E) {
                val x1 = boxArray[0]; val y1 = boxArray[1]
                val x2 = boxArray[2]; val y2 = boxArray[3]
                mapLetterboxE2EToOriginal(x1, y1, x2, y2, origWidth, origHeight, lb)
            } else {
                // JNI: [x, y, width, height] normalized to letterboxed input tensor (xywh, top-left + size).
                val x = boxArray[0]; val y = boxArray[1]
                val w = boxArray[2]; val h = boxArray[3]
                mapLetterboxNormXywhToOriginal(x, y, w, h, origWidth, origHeight, lb)
            }

            val left = rect.left
            val top = rect.top
            val right = rect.right
            val bottom = rect.bottom
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
    // Defaults aligned with Ultralytics COCO-style predict (conf=0.001, iou NMS=0.7, max_det=300).
    private var confidenceThreshold = 0.001f
    private var iouThreshold = 0.7f

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
        init {
            System.loadLibrary("ultralytics")
        }
    }
}