import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:ultralytics_yolo/yolo.dart';
import '../../models/models.dart';
import '../../models/benchmark_result.dart';
import '../../services/model_manager.dart';

class BenchmarkController extends ChangeNotifier {
  static const benchmarkModels = [
    ModelType.customFloat32,
    ModelType.customFloat16,
    ModelType.customInt8,
  ];

  ModelType _selectedModel = ModelType.customFloat32;
  List<XFile> _selectedImages = [];
  bool _isModelLoading = false;
  bool _isRunning = false;
  int _processedCount = 0;
  List<BenchmarkResult> _results = [];
  String? _errorMessage;
  String? _lastCsvPath;
  String? _lastPredictError;
  int _failedCount = 0;
  YOLO? _yolo;

  ModelType get selectedModel => _selectedModel;
  List<XFile> get selectedImages => _selectedImages;
  bool get isModelLoading => _isModelLoading;
  bool get isRunning => _isRunning;
  int get processedCount => _processedCount;
  List<BenchmarkResult> get results => _results;
  String? get errorMessage => _errorMessage;
  String? get lastPredictError => _lastPredictError;
  int get failedCount => _failedCount;
  bool get hasResults => _results.isNotEmpty;
  bool get canRun => !_isModelLoading && !_isRunning && _selectedImages.isNotEmpty && _yolo != null;

  BenchmarkSummary get summary => BenchmarkSummary.fromResults(_results);

  Future<void> selectModel(ModelType model) async {
    _selectedModel = model;
    _results = [];
    _lastCsvPath = null;
    notifyListeners();
    await _loadModel();
  }

  Future<void> _loadModel() async {
    _isModelLoading = true;
    _errorMessage = null;
    notifyListeners();

    try {
      _yolo = null;
      final modelManager = ModelManager();
      final path = await modelManager.getModelPath(_selectedModel);
      if (path == null) {
        _errorMessage = 'Modelo ${_selectedModel.modelName} nao encontrado';
        _isModelLoading = false;
        notifyListeners();
        return;
      }
      _yolo = YOLO(modelPath: path, task: _selectedModel.task, useGpu: false);
      await _yolo!.loadModel();
    } catch (e) {
      _errorMessage = 'Erro ao carregar modelo: $e';
      _yolo = null;
    }

    _isModelLoading = false;
    notifyListeners();
  }

  void setImages(List<XFile> images) {
    _selectedImages = images;
    _results = [];
    _lastCsvPath = null;
    notifyListeners();
  }

  Future<void> runBenchmark() async {
    if (_yolo == null || _selectedImages.isEmpty) return;

    _isRunning = true;
    _processedCount = 0;
    _results = [];
    _lastCsvPath = null;
    _errorMessage = null;
    _lastPredictError = null;
    _failedCount = 0;
    notifyListeners();

    for (final image in _selectedImages) {
      try {
        final bytes = await image.readAsBytes();
        final sw = Stopwatch()..start();
        final result = await _yolo!.predict(bytes);
        sw.stop();

        final timeMs = sw.elapsedMicroseconds / 1000.0;
        final boxes = result['boxes'];
        final detections = boxes is List ? boxes.length : 0;

        _results.add(BenchmarkResult(
          imageName: image.name,
          inferenceTimeMs: timeMs,
          detectionCount: detections,
        ));
      } catch (e) {
        _lastPredictError = e.toString();
        _failedCount++;
      }

      _processedCount++;
      notifyListeners();
    }

    _isRunning = false;
    notifyListeners();
  }

  Future<String> exportCsv() async {
    if (_lastCsvPath != null) return _lastCsvPath!;

    final buffer = StringBuffer();
    buffer.writeln('image,inference_time_ms,detections');
    for (final r in _results) {
      buffer.writeln('${r.imageName},${r.inferenceTimeMs.toStringAsFixed(2)},${r.detectionCount}');
    }

    final s = summary;
    buffer.writeln();
    buffer.writeln('SUMMARY');
    buffer.writeln('model,images,avg_ms,min_ms,max_ms,fps,total_detections,avg_detections');
    buffer.writeln(
      '${_selectedModel.modelName},${s.imageCount},'
      '${s.avgTimeMs.toStringAsFixed(2)},${s.minTimeMs.toStringAsFixed(2)},'
      '${s.maxTimeMs.toStringAsFixed(2)},${s.fps.toStringAsFixed(2)},'
      '${s.totalDetections},${s.avgDetectionsPerImage.toStringAsFixed(2)}',
    );

    final dir = await getApplicationDocumentsDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final file = File('${dir.path}/benchmark_${_selectedModel.modelName}_$timestamp.csv');
    await file.writeAsString(buffer.toString());

    _lastCsvPath = file.path;
    return file.path;
  }

  Future<void> shareCsv() async {
    final path = await exportCsv();
    await Share.shareXFiles([XFile(path)], text: 'Benchmark ${_selectedModel.modelName}');
  }

  Future<void> initialize() async {
    await _loadModel();
  }

  @override
  void dispose() {
    _yolo = null;
    super.dispose();
  }
}
