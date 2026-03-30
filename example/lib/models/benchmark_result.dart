class BenchmarkResult {
  final String imageName;
  final double inferenceTimeMs;
  final int detectionCount;

  const BenchmarkResult({
    required this.imageName,
    required this.inferenceTimeMs,
    required this.detectionCount,
  });
}

class BenchmarkSummary {
  final double avgTimeMs;
  final double minTimeMs;
  final double maxTimeMs;
  final double fps;
  final int totalDetections;
  final double avgDetectionsPerImage;
  final int imageCount;

  const BenchmarkSummary({
    required this.avgTimeMs,
    required this.minTimeMs,
    required this.maxTimeMs,
    required this.fps,
    required this.totalDetections,
    required this.avgDetectionsPerImage,
    required this.imageCount,
  });

  factory BenchmarkSummary.fromResults(List<BenchmarkResult> results) {
    if (results.isEmpty) {
      return const BenchmarkSummary(
        avgTimeMs: 0,
        minTimeMs: 0,
        maxTimeMs: 0,
        fps: 0,
        totalDetections: 0,
        avgDetectionsPerImage: 0,
        imageCount: 0,
      );
    }

    final times = results.map((r) => r.inferenceTimeMs).toList();
    final totalDetections =
        results.fold<int>(0, (sum, r) => sum + r.detectionCount);
    final avgTime = times.reduce((a, b) => a + b) / times.length;

    return BenchmarkSummary(
      avgTimeMs: avgTime,
      minTimeMs: times.reduce((a, b) => a < b ? a : b),
      maxTimeMs: times.reduce((a, b) => a > b ? a : b),
      fps: avgTime > 0 ? 1000.0 / avgTime : 0,
      totalDetections: totalDetections,
      avgDetectionsPerImage: totalDetections / results.length,
      imageCount: results.length,
    );
  }
}
