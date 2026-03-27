import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../../models/models.dart';
import '../controllers/benchmark_controller.dart';

class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({super.key});

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  late final BenchmarkController _controller;
  final _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _controller = BenchmarkController();
    _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _pickImages() async {
    final images = await _picker.pickMultiImage();
    if (images.isNotEmpty) {
      _controller.setImages(images);
    }
  }

  void _showSnackBar(String msg) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Benchmark'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: ListenableBuilder(
        listenable: _controller,
        builder: (context, _) {
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _buildModelSelector(),
                const SizedBox(height: 16),
                _buildImageSelector(),
                const SizedBox(height: 16),
                _buildRunButton(),
                if (_controller.isRunning) ...[
                  const SizedBox(height: 16),
                  _buildProgress(),
                ],
                if (_controller.errorMessage != null) ...[
                  const SizedBox(height: 16),
                  Text(
                    _controller.errorMessage!,
                    style: const TextStyle(color: Colors.red),
                  ),
                ],
                if (_controller.lastPredictError != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    'Erro na inferencia: ${_controller.lastPredictError}',
                    style: const TextStyle(color: Colors.orange, fontSize: 12),
                  ),
                ],
                if (_controller.hasResults && !_controller.isRunning) ...[
                  const SizedBox(height: 24),
                  _buildResults(),
                  const SizedBox(height: 16),
                  _buildExportButtons(),
                ],
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildModelSelector() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Modelo', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            const SizedBox(height: 8),
            DropdownButtonFormField<ModelType>(
              initialValue: _controller.selectedModel,
              isExpanded: true,
              items: BenchmarkController.benchmarkModels.map((model) {
                return DropdownMenuItem(
                  value: model,
                  child: Text(model.modelName),
                );
              }).toList(),
              onChanged: _controller.isRunning
                  ? null
                  : (model) {
                      if (model != null) _controller.selectModel(model);
                    },
            ),
            if (_controller.isModelLoading)
              const Padding(
                padding: EdgeInsets.only(top: 8),
                child: Row(
                  children: [
                    SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
                    SizedBox(width: 8),
                    Text('Carregando modelo...'),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildImageSelector() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Imagens', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            const SizedBox(height: 8),
            Row(
              children: [
                ElevatedButton.icon(
                  onPressed: _controller.isRunning ? null : _pickImages,
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Selecionar da galeria'),
                ),
                const SizedBox(width: 12),
                Text(
                  '${_controller.selectedImages.length} selecionadas',
                  style: TextStyle(color: Colors.grey[600]),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRunButton() {
    return SizedBox(
      height: 48,
      child: ElevatedButton.icon(
        onPressed: _controller.canRun ? () => _controller.runBenchmark() : null,
        icon: const Icon(Icons.play_arrow),
        label: const Text('Rodar Benchmark', style: TextStyle(fontSize: 16)),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.green,
          foregroundColor: Colors.white,
        ),
      ),
    );
  }

  Widget _buildProgress() {
    final total = _controller.selectedImages.length;
    final processed = _controller.processedCount;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            LinearProgressIndicator(value: total > 0 ? processed / total : 0),
            const SizedBox(height: 8),
            Text('Processando imagem $processed de $total...'),
          ],
        ),
      ),
    );
  }

  Widget _buildResults() {
    final s = _controller.summary;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Resultados - ${_controller.selectedModel.modelName}',
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
            ),
            const Divider(),
            _metricRow('Imagens processadas', '${s.imageCount}'),
            if (_controller.failedCount > 0)
              _metricRow('Falhas', '${_controller.failedCount}', valueColor: Colors.red),
            _metricRow('Tempo medio', '${s.avgTimeMs.toStringAsFixed(1)} ms'),
            _metricRow('Tempo minimo', '${s.minTimeMs.toStringAsFixed(1)} ms'),
            _metricRow('Tempo maximo', '${s.maxTimeMs.toStringAsFixed(1)} ms'),
            _metricRow('FPS equivalente', s.fps.toStringAsFixed(1)),
            const Divider(),
            _metricRow('Total de deteccoes', '${s.totalDetections}'),
            _metricRow('Media deteccoes/imagem', s.avgDetectionsPerImage.toStringAsFixed(1)),
          ],
        ),
      ),
    );
  }

  Widget _metricRow(String label, String value, {Color? valueColor}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(color: Colors.grey[700])),
          Text(value, style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: valueColor)),
        ],
      ),
    );
  }

  Widget _buildExportButtons() {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: () async {
              try {
                final path = await _controller.exportCsv();
                _showSnackBar('CSV salvo em: $path');
              } catch (e) {
                _showSnackBar('Erro ao salvar: $e');
              }
            },
            icon: const Icon(Icons.save),
            label: const Text('Salvar CSV'),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: () async {
              try {
                await _controller.shareCsv();
              } catch (e) {
                _showSnackBar('Erro ao compartilhar: $e');
              }
            },
            icon: const Icon(Icons.share),
            label: const Text('Compartilhar'),
          ),
        ),
      ],
    );
  }
}
