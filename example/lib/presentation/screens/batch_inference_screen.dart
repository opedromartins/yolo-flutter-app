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
import '../../models/models.dart';
import '../../services/model_manager.dart';

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
  String _status = 'Carregando modelo EPI...';
  double _progress = 0.0;
  bool _isRunning = false;

  final List<Map<String, dynamic>> _results = [];
  String? _outputPath;
  int _processedCount = 0;
  int _totalCount = 0;

  /// Resumo do último lote: tempo só de `predict` (ms) e FPS equivalente (1000/avg).
  Map<String, double>? _timingSummary;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      final modelPath = await _modelManager.getModelPath(ModelType.detect);
      if (modelPath == null || !mounted) return;

      _yolo = YOLO(
        modelPath: modelPath,
        task: YOLOTask.detect,
        useGpu: !Platform.isAndroid,
      );
      await _yolo!.loadModel();

      if (mounted) {
        setState(() {
          _isModelReady = true;
          _status = 'Modelo pronto. Selecione as imagens do dataset valid.';
        });
      }
    } catch (e) {
      if (mounted) {
        final error = YOLOErrorHandler.handleError(e, 'Falha ao carregar modelo');
        setState(() => _status = 'Erro: ${error.message}');
      }
    }
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

      await _runInferenceOnPaths(paths);
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
      await _runInferenceOnPaths(imageFiles.map((f) => f.path).toList());
    } catch (e) {
      if (mounted) _showSnackBar('Erro ao acessar pasta: $e');
    }
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

  Future<void> _runInferenceOnPaths(List<String> filePaths) async {
    if (!mounted || _yolo == null) return;

    setState(() {
      _isRunning = true;
      _results.clear();
      _timingSummary = null;
      _totalCount = filePaths.length;
      _processedCount = 0;
      _progress = 0.0;
    });

    final allResults = <Map<String, dynamic>>[];
    final inferenceTimesMs = <double>[];
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

          allResults.add({
            'file': file.path.split(RegExp(r'[/\\]')).last,
            'path': file.path,
            'detections': boxes.length,
            'inference_time_ms': inferenceMs,
            'boxes': boxes,
          });
        } catch (e) {
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

      if (mounted && allResults.isNotEmpty) {
        final payload = <String, dynamic>{
          'results': allResults,
          if (summary != null)
            'summary': {
              ...summary,
              'images_timed': inferenceTimesMs.length,
              'images_total': allResults.length,
            },
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
          _status =
              'Concluído! ${allResults.length} imagens processadas$avgStr.';
          _isRunning = false;
        });
        _showSnackBar('Resultados salvos');
      } else if (mounted) {
        setState(() {
          _timingSummary = summary;
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
                        'O arquivo está na pasta interna do app (inacessível). '
                        'Use o botão abaixo para compartilhar e salvar em um local visível.',
                        style: TextStyle(fontSize: 12),
                      ),
                      const SizedBox(height: 12),
                      ElevatedButton.icon(
                        onPressed: _shareOrSaveResults,
                        icon: const Icon(Icons.share),
                        label: const Text('Compartilhar / Salvar em Download'),
                      ),
                    ],
                  ),
                ),
              ),
            ],
            if (_results.isNotEmpty) ...[
              const SizedBox(height: 16),
              Expanded(
                child: ListView.builder(
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
              ),
            ],
          ],
        ),
      ),
    );
  }
}
