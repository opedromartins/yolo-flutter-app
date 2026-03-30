// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:convert';
import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:saf/saf.dart';
import 'package:share_plus/share_plus.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:ultralytics_yolo/utils/map_converter.dart';
import 'package:ultralytics_yolo/utils/error_handler.dart';
import '../../models/benchmark_result.dart';
import '../../models/models.dart';
import '../../services/model_manager.dart';
import '../../utils/detection_eval_metrics.dart';
import '../controllers/benchmark_controller.dart';

/// Inferência em lote no dataset valid com o modelo EPI customizado.
///
/// 1. Copie a pasta valid para o celular (via USB para Download)
/// 2. Toque em "Selecionar imagens" e navegue até a pasta
/// 3. Selecione todas as imagens do dataset
class BatchInferenceScreen extends StatefulWidget {
  const BatchInferenceScreen({super.key});

  @override
  State<BatchInferenceScreen> createState() => _BatchInferenceScreenState();
}

class _BatchInferenceScreenState extends State<BatchInferenceScreen> {
  final _modelManager = ModelManager();
  YOLO? _yolo;
  bool _isModelReady = false;
  bool _isModelLoading = false;
  ModelType _selectedModel = BenchmarkController.benchmarkModels.first;
  String _status = 'Carregando modelo...';
  double _progress = 0.0;
  bool _isRunning = false;

  final List<Map<String, dynamic>> _results = [];
  String? _outputPath;
  int _processedCount = 0;
  int _totalCount = 0;

  /// Resumo do último lote: tempo só de `predict` (ms) e FPS equivalente (1000/avg).
  Map<String, double>? _timingSummary;

  /// Igual ao benchmark: lista de tempos por imagem bem-sucedida.
  List<BenchmarkResult> _benchmarkResults = [];
  BenchmarkSummary? _benchmarkSummary;
  int _failedInferenceCount = 0;

  /// Calculadas para JSON/CSV export (não mostradas na UI).
  Map<String, dynamic>? _accuracyMetrics;

  String? _lastCsvPath;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    setState(() {
      _isModelLoading = true;
      _isModelReady = false;
      _status = 'Carregando ${_selectedModel.modelName}...';
    });
    try {
      _yolo = null;
      final modelPath = await _modelManager.getModelPath(_selectedModel);
      if (modelPath == null || !mounted) {
        setState(() {
          _isModelLoading = false;
          _status = 'Modelo ${_selectedModel.modelName} não encontrado.';
        });
        return;
      }

      _yolo = YOLO(
        modelPath: modelPath,
        task: _selectedModel.task,
        useGpu: !Platform.isAndroid,
      );
      await _yolo!.loadModel();

      if (mounted) {
        setState(() {
          _isModelLoading = false;
          _isModelReady = true;
          _status =
              'Modelo ${_selectedModel.modelName} pronto. Selecione as imagens.';
        });
      }
    } catch (e) {
      if (mounted) {
        final error = YOLOErrorHandler.handleError(e, 'Falha ao carregar modelo');
        setState(() {
          _isModelLoading = false;
          _isModelReady = false;
          _status = 'Erro: ${error.message}';
        });
      }
    }
  }

  Future<void> _onModelChanged(ModelType? model) async {
    if (model == null || _isRunning || model == _selectedModel) return;
    setState(() => _selectedModel = model);
    await _loadModel();
  }

  Future<void> _pickFolderAndRunInference() async {
    if (!_isModelReady || _yolo == null || _isRunning) return;

    if (Platform.isAndroid) {
      await _pickFolderWithSaf();
    } else {
      await _pickFolderWithFilePicker();
    }
  }

  /// Android: usa SAF - cache() copia imagens para o app (paths legíveis)
  Future<void> _pickFolderWithSaf() async {
    try {
      final saf = Saf('~');
      final granted = await saf.getDirectoryPermission(isDynamic: true);
      if (granted != true || !mounted) return;

      setState(() => _status = 'Copiando arquivos da pasta...');

      // Usa 'any' e filtra imagens (fileType image pode ser restritivo demais)
      final allPaths = await saf.cache(fileType: FileTypes.any);
      if (allPaths == null || allPaths.isEmpty || !mounted) {
        _showSnackBar('Nenhum arquivo encontrado na pasta');
        return;
      }

      final paths = allPaths.where((p) {
        final lower = p.toLowerCase();
        return lower.endsWith('.jpg') || lower.endsWith('.jpeg') ||
            lower.endsWith('.png') || lower.endsWith('.bmp') ||
            lower.endsWith('.webp') || lower.endsWith('.gif');
      }).toList();

      if (paths.isEmpty || !mounted) {
        _showSnackBar('Nenhuma imagem (.jpg, .png) encontrada na pasta');
        return;
      }

      // SAF copia tudo para cache: incluir .txt do mesmo lote (labels em paralelo a images).
      final labelMap = await loadYoloLabelsFromPathList(allPaths);
      await _runInferenceOnPaths(paths, batchLabelMap: labelMap);
    } catch (e) {
      if (mounted) _showSnackBar('Erro: $e');
    }
  }

  /// iOS/outros: usa file_picker (pode não listar em algumas versões)
  Future<void> _pickFolderWithFilePicker() async {
    final dirPath = await FilePicker.platform.getDirectoryPath(
      dialogTitle: 'Selecione a pasta do dataset',
    );
    if (dirPath == null || dirPath.isEmpty || !mounted) return;

    try {
      final dir = Directory(dirPath);
      if (!await dir.exists()) {
        _showSnackBar('Pasta não encontrada');
        return;
      }
      final imageFiles = <File>[];
      await for (final entity in dir.list(recursive: true)) {
        if (entity is File) {
          final lower = entity.path.toLowerCase();
          if (lower.endsWith('.jpg') || lower.endsWith('.jpeg') ||
              lower.endsWith('.png') || lower.endsWith('.bmp') ||
              lower.endsWith('.webp') || lower.endsWith('.gif')) {
            imageFiles.add(entity);
          }
        }
      }
      if (imageFiles.isEmpty) {
        _showSnackBar('Nenhuma imagem encontrada na pasta');
        return;
      }
      final txtPaths = await _collectTxtPathsUnder(dirPath);
      final labelMap = await loadYoloLabelsFromPathList(txtPaths);
      await _runInferenceOnPaths(
        imageFiles.map((f) => f.path).toList(),
        batchLabelMap: labelMap,
      );
    } catch (e) {
      if (mounted) _showSnackBar('Erro ao acessar pasta: $e');
    }
  }

  Future<List<String>> _collectTxtPathsUnder(String dirPath) async {
    final out = <String>[];
    await for (final entity in Directory(dirPath).list(recursive: true)) {
      if (entity is File && entity.path.toLowerCase().endsWith('.txt')) {
        out.add(entity.path);
      }
    }
    return out;
  }

  Future<void> _pickAndRunInference() async {
    if (!_isModelReady || _yolo == null || _isRunning) return;

    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: true,
    );

    if (result == null || result.files.isEmpty || !mounted) return;

    final files = result.files.where((f) => f.path != null).toList();
    if (files.isEmpty) {
      _showSnackBar('Nenhum arquivo selecionado');
      return;
    }

    final paths = files
        .where((f) => f.path != null && f.path!.isNotEmpty)
        .map((f) => f.path!)
        .toList();
    if (paths.isEmpty) {
      _showSnackBar('Nenhum arquivo válido');
      return;
    }
    await _runInferenceOnPaths(paths);
  }

  Future<void> _runInferenceOnPaths(
    List<String> filePaths, {
    Map<String, List<GtBox>>? batchLabelMap,
  }) async {
    if (!mounted || _yolo == null) return;

    setState(() {
      _isRunning = true;
      _results.clear();
      _timingSummary = null;
      _benchmarkResults = [];
      _benchmarkSummary = null;
      _failedInferenceCount = 0;
      _accuracyMetrics = null;
      _lastCsvPath = null;
      _totalCount = filePaths.length;
      _processedCount = 0;
      _progress = 0.0;
    });

    final allResults = <Map<String, dynamic>>[];
    final inferenceTimesMs = <double>[];
    final benchmarkRows = <BenchmarkResult>[];
    final gtAll = <String, List<GtBox>>{};
    if (batchLabelMap != null && batchLabelMap.isNotEmpty) {
      gtAll.addAll(batchLabelMap);
    }
    var failedInferenceCount = 0;
    final outputDir = await getApplicationDocumentsDirectory();
    final timestamp =
        DateTime.now().toIso8601String().replaceAll(':', '-').split('.').first;
    final resultsFile =
        File('${outputDir.path}/inference_valid_$timestamp.json');

    try {
      for (int i = 0; i < filePaths.length; i++) {
        if (!mounted) break;

        final file = File(filePaths[i]);
        if (!await file.exists()) continue;

        setState(() {
          _status =
              'Processando ${i + 1}/${filePaths.length}: ${file.path.split(RegExp(r'[/\\]')).last}';
          _progress = (i + 1) / filePaths.length;
        });

        final stem = baseName(file.path.split(RegExp(r'[/\\]')).last);
        if (!gtAll.containsKey(stem)) {
          final labelPath = resolveYoloLabelPath(file.path);
          if (labelPath != null && await File(labelPath).exists()) {
            gtAll[stem] = await loadYoloLabelFile(labelPath);
          }
        }

        try {
          final bytes = await file.readAsBytes();
          final sw = Stopwatch()..start();
          final predictResult = await _yolo!.predict(bytes);
          sw.stop();
          final inferenceMs = sw.elapsedMicroseconds / 1000.0;
          inferenceTimesMs.add(inferenceMs);

          final boxes = predictResult['boxes'] is List
              ? MapConverter.convertBoxesList(predictResult['boxes'] as List)
              : <Map<String, dynamic>>[];

          final fileName = file.path.split(RegExp(r'[/\\]')).last;
          allResults.add({
            'file': fileName,
            'path': file.path,
            'detections': boxes.length,
            'inference_time_ms': inferenceMs,
            'boxes': boxes,
          });
          benchmarkRows.add(BenchmarkResult(
            imageName: fileName,
            inferenceTimeMs: inferenceMs,
            detectionCount: boxes.length,
          ));
        } catch (e) {
          failedInferenceCount++;
          allResults.add({
            'file': file.path.split(RegExp(r'[/\\]')).last,
            'path': file.path,
            'error': e.toString(),
          });
        }

        setState(() => _processedCount = i + 1);
      }

      Map<String, double>? summary;
      if (inferenceTimesMs.isNotEmpty) {
        final sum = inferenceTimesMs.reduce((a, b) => a + b);
        final avg = sum / inferenceTimesMs.length;
        summary = {
          'avg_ms': avg,
          'min_ms': inferenceTimesMs.reduce((a, b) => a < b ? a : b),
          'max_ms': inferenceTimesMs.reduce((a, b) => a > b ? a : b),
          'fps': 1000.0 / avg,
        };
      }

      final summaryBenchmark = BenchmarkSummary.fromResults(benchmarkRows);
      Map<String, dynamic>? accMetrics;
      final gtBoxCount =
          gtAll.values.fold<int>(0, (s, g) => s + g.length);
      if (gtBoxCount > 0) {
        accMetrics = computeDetectionMetrics(allResults, gtAll);
      }

      if (mounted && allResults.isNotEmpty) {
        final payload = <String, dynamic>{
          'model': _selectedModel.modelName,
          'results': allResults,
          if (summary != null)
            'summary': {
              ...summary,
              'images_timed': inferenceTimesMs.length,
              'images_total': allResults.length,
            },
          'benchmark_summary': {
            'model': _selectedModel.modelName,
            'avg_ms': summaryBenchmark.avgTimeMs,
            'min_ms': summaryBenchmark.minTimeMs,
            'max_ms': summaryBenchmark.maxTimeMs,
            'fps': summaryBenchmark.fps,
            'image_count': summaryBenchmark.imageCount,
            'total_detections': summaryBenchmark.totalDetections,
            'avg_detections_per_image': summaryBenchmark.avgDetectionsPerImage,
            'failed_inference_count': failedInferenceCount,
          },
          if (accMetrics != null) 'accuracy_metrics': accMetrics,
        };
        await resultsFile.writeAsString(
          const JsonEncoder.withIndent('  ').convert(payload),
          encoding: utf8,
        );
        final avgStr = summary != null
            ? ' | média ${summary['avg_ms']!.toStringAsFixed(1)} ms · '
                '${summary['fps']!.toStringAsFixed(1)} FPS'
            : '';
        setState(() {
          _outputPath = resultsFile.path;
          _results.addAll(allResults);
          _timingSummary = summary;
          _benchmarkResults = benchmarkRows;
          _benchmarkSummary = summaryBenchmark;
          _accuracyMetrics = accMetrics;
          _failedInferenceCount = failedInferenceCount;
          _status =
              'Concluído! ${allResults.length} imagens processadas$avgStr.';
          _isRunning = false;
        });
        _showSnackBar('Resultados salvos');
      } else if (mounted) {
        setState(() {
          _timingSummary = summary;
          _benchmarkResults = benchmarkRows;
          _benchmarkSummary = BenchmarkSummary.fromResults(benchmarkRows);
          _accuracyMetrics = accMetrics;
          _failedInferenceCount = failedInferenceCount;
          _status = allResults.isEmpty
              ? 'Nenhuma imagem processada.'
              : 'Concluído sem arquivo (lista vazia).';
          _isRunning = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _status = 'Erro: $e';
          _isRunning = false;
        });
      }
    }
  }

  void _showSnackBar(String msg) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
    }
  }

  Widget _metricRow(String label, String value, {Color? valueColor}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(color: Colors.grey[700])),
          Text(
            value,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 15,
              color: valueColor,
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _shareOrSaveResults() async {
    if (_outputPath == null) return;
    final file = File(_outputPath!);
    if (!await file.exists()) {
      _showSnackBar('Arquivo não encontrado');
      return;
    }
    try {
      final bytes = await file.readAsBytes();
      final xFile = XFile.fromData(
        bytes,
        name: 'inference_valid_${DateTime.now().millisecondsSinceEpoch}.json',
        mimeType: 'application/json',
      );
      await Share.shareXFiles(
        [xFile],
        text: 'Resultados inferência dataset valid',
      );
    } catch (e) {
      _showSnackBar('Erro ao compartilhar: $e');
    }
  }

  /// JSON só com `model` + `results` (file, detecções, boxes, tempo; sem summary/benchmark no topo).
  Map<String, dynamic> _detectionsOnlyPayload() {
    final results = _results.map((r) {
      if (r['error'] != null) {
        return <String, dynamic>{
          'file': r['file'],
          'error': r['error'].toString(),
        };
      }
      return <String, dynamic>{
        'file': r['file'],
        'detections': r['detections'],
        'inference_time_ms': r['inference_time_ms'],
        'boxes': r['boxes'],
      };
    }).toList();
    return {
      'model': _selectedModel.modelName,
      'results': results,
    };
  }

  Future<void> _shareDetectionsJsonOnly() async {
    if (_results.isEmpty) {
      _showSnackBar('Nenhum resultado para exportar');
      return;
    }
    try {
      final json = const JsonEncoder.withIndent('  ').convert(
        _detectionsOnlyPayload(),
      );
      final dir = await getApplicationDocumentsDirectory();
      final ts = DateTime.now().millisecondsSinceEpoch;
      final f = File('${dir.path}/detections_only_$ts.json');
      await f.writeAsString(json, encoding: utf8);
      await Share.shareXFiles(
        [XFile(f.path)],
        text: 'Detecções YOLO (JSON)',
      );
    } catch (e) {
      _showSnackBar('Erro ao exportar JSON: $e');
    }
  }

  Future<String> _exportBenchmarkCsv() async {
    if (_lastCsvPath != null) return _lastCsvPath!;
    final buffer = StringBuffer();
    buffer.writeln('image,inference_time_ms,detections');
    for (final r in _benchmarkResults) {
      buffer.writeln(
        '${r.imageName},${r.inferenceTimeMs.toStringAsFixed(2)},${r.detectionCount}',
      );
    }
    final s = _benchmarkSummary ?? BenchmarkSummary.fromResults(_benchmarkResults);
    buffer.writeln();
    buffer.writeln('SUMMARY');
    buffer.writeln(
      'model,images,avg_ms,min_ms,max_ms,fps,total_detections,avg_detections,failed_inference,'
      'precision,recall,f1,accuracy,map50,tp,fp,fn',
    );
    final m = _accuracyMetrics;
    final accCsv = m == null
        ? List<String>.filled(9, '')
        : <String>[
            (m['precision'] as num).toStringAsFixed(4),
            (m['recall'] as num).toStringAsFixed(4),
            (m['f1'] as num).toStringAsFixed(4),
            (m['accuracy'] as num).toStringAsFixed(4),
            (m['mAP@0.5'] as num).toStringAsFixed(4),
            '${m['tp']}',
            '${m['fp']}',
            '${m['fn']}',
          ];
    buffer.writeln(
      '${_selectedModel.modelName},${s.imageCount},'
      '${s.avgTimeMs.toStringAsFixed(2)},${s.minTimeMs.toStringAsFixed(2)},'
      '${s.maxTimeMs.toStringAsFixed(2)},${s.fps.toStringAsFixed(2)},'
      '${s.totalDetections},${s.avgDetectionsPerImage.toStringAsFixed(2)},'
      '$_failedInferenceCount,'
      '${accCsv.join(',')}',
    );
    final dir = await getApplicationDocumentsDirectory();
    final ts = DateTime.now().millisecondsSinceEpoch;
    final file = File(
      '${dir.path}/batch_${_selectedModel.modelName}_$ts.csv',
    );
    await file.writeAsString(buffer.toString());
    _lastCsvPath = file.path;
    return file.path;
  }

  Future<void> _shareBenchmarkCsv() async {
    final path = await _exportBenchmarkCsv();
    await Share.shareXFiles(
      [XFile(path)],
      text: 'Batch ${_selectedModel.modelName}',
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Dataset Valid - Inferência em Lote'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(_status,
                        style: Theme.of(context).textTheme.bodyLarge),
                    if (_timingSummary != null && !_isRunning) ...[
                      const SizedBox(height: 10),
                      Text(
                        'Tempo de inferência (predict): '
                        'média ${_timingSummary!['avg_ms']!.toStringAsFixed(1)} ms · '
                        'min ${_timingSummary!['min_ms']!.toStringAsFixed(1)} · '
                        'max ${_timingSummary!['max_ms']!.toStringAsFixed(1)} · '
                        '${_timingSummary!['fps']!.toStringAsFixed(1)} FPS',
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.blueGrey.shade800,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                    if (_isRunning) ...[
                      const SizedBox(height: 12),
                      LinearProgressIndicator(value: _progress),
                      const SizedBox(height: 8),
                      Text('$_processedCount / $_totalCount imagens'),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Modelo',
                      style:
                          TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                    ),
                    const SizedBox(height: 8),
                    DropdownButtonFormField<ModelType>(
                      initialValue: _selectedModel,
                      isExpanded: true,
                      items: BenchmarkController.benchmarkModels.map((model) {
                        return DropdownMenuItem(
                          value: model,
                          child: Text(model.modelName),
                        );
                      }).toList(),
                      onChanged: _isRunning || _isModelLoading
                          ? null
                          : _onModelChanged,
                    ),
                    if (_isModelLoading)
                      const Padding(
                        padding: EdgeInsets.only(top: 8),
                        child: Row(
                          children: [
                            SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            ),
                            SizedBox(width: 8),
                            Text('Carregando modelo...'),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: _isModelReady && !_isRunning ? _pickFolderAndRunInference : null,
              icon: const Icon(Icons.folder),
              label: const Text('Selecionar pasta'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: _isModelReady && !_isRunning ? _pickAndRunInference : null,
              icon: const Icon(Icons.photo_library),
              label: const Text('Selecionar imagens da galeria'),
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),
            const SizedBox(height: 16),
            const Text(
              'IMPORTANTE: Selecione a pasta onde estão as imagens .jpg.\n'
              'Ex: se as imagens estão em valid/images/, selecione "images".\n'
              'Não selecione a pasta pai (valid) - entre dentro dela primeiro.',
              style: TextStyle(fontSize: 12, color: Colors.grey),
            ),
            if (_benchmarkSummary != null && !_isRunning) ...[
              const SizedBox(height: 16),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Benchmark — ${_selectedModel.modelName}',
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 18,
                        ),
                      ),
                      const Divider(),
                      _metricRow(
                        'Imagens com inferência OK',
                        '${_benchmarkSummary!.imageCount}',
                      ),
                      if (_failedInferenceCount > 0)
                        _metricRow(
                          'Falhas na inferência',
                          '$_failedInferenceCount',
                          valueColor: Colors.red,
                        ),
                      _metricRow(
                        'Tempo médio',
                        '${_benchmarkSummary!.avgTimeMs.toStringAsFixed(1)} ms',
                      ),
                      _metricRow(
                        'Tempo mínimo',
                        '${_benchmarkSummary!.minTimeMs.toStringAsFixed(1)} ms',
                      ),
                      _metricRow(
                        'Tempo máximo',
                        '${_benchmarkSummary!.maxTimeMs.toStringAsFixed(1)} ms',
                      ),
                      _metricRow(
                        'FPS equivalente',
                        _benchmarkSummary!.fps.toStringAsFixed(1),
                      ),
                      const Divider(),
                      _metricRow(
                        'Total de detecções',
                        '${_benchmarkSummary!.totalDetections}',
                      ),
                      _metricRow(
                        'Média detecções/imagem',
                        _benchmarkSummary!.avgDetectionsPerImage.toStringAsFixed(1),
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: _benchmarkResults.isEmpty
                                  ? null
                                  : () async {
                                      try {
                                        _lastCsvPath = null;
                                        final path = await _exportBenchmarkCsv();
                                        _showSnackBar('CSV salvo em: $path');
                                      } catch (e) {
                                        _showSnackBar('Erro: $e');
                                      }
                                    },
                              icon: const Icon(Icons.save),
                              label: const Text('Salvar CSV'),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: _benchmarkResults.isEmpty
                                  ? null
                                  : () async {
                                      try {
                                        _lastCsvPath = null;
                                        await _shareBenchmarkCsv();
                                      } catch (e) {
                                        _showSnackBar('Erro: $e');
                                      }
                                    },
                              icon: const Icon(Icons.share),
                              label: const Text('CSV'),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
            if (_outputPath != null) ...[
              const SizedBox(height: 24),
              Card(
                color: Colors.green.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Resultados salvos.',
                          style: TextStyle(fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      const Text(
                        'O arquivo completo está na pasta interna do app. '
                        'Use os botões para partilhar (ex.: Guardar no telefone / Drive).',
                        style: TextStyle(fontSize: 12),
                      ),
                      const SizedBox(height: 12),
                      ElevatedButton.icon(
                        onPressed: _shareOrSaveResults,
                        icon: const Icon(Icons.description),
                        label: const Text('JSON completo (inferência + resumos)'),
                      ),
                      const SizedBox(height: 8),
                      OutlinedButton.icon(
                        onPressed: _shareDetectionsJsonOnly,
                        icon: const Icon(Icons.image_search),
                        label: const Text('JSON só detecções (boxes)'),
                      ),
                    ],
                  ),
                ),
              ),
            ],
            if (_results.isNotEmpty) ...[
              const SizedBox(height: 16),
              ListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _results.length,
                itemBuilder: (context, i) {
                  final r = _results[i];
                  final detections = r['detections'] ?? 0;
                  final error = r['error'];
                  return ListTile(
                    dense: true,
                    title: Text(r['file'] ?? 'Imagem ${i + 1}',
                        overflow: TextOverflow.ellipsis),
                    subtitle: Text(
                      error != null
                          ? 'Erro'
                          : () {
                              final ms = r['inference_time_ms'];
                              final msStr = ms is num
                                  ? ms.toDouble().toStringAsFixed(1)
                                  : '?';
                              return '$detections detecções · $msStr ms';
                            }(),
                    ),
                    trailing: Icon(
                      error != null ? Icons.error : Icons.check_circle,
                      color: error != null ? Colors.red : Colors.green,
                      size: 20,
                    ),
                  );
                },
              ),
            ],
          ],
        ),
      ),
      ),
    );
  }
}
