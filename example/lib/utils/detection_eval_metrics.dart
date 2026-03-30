// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'dart:math' as math;

/// Ground truth box: YOLO normalized (class, xc, yc, w, h).
class GtBox {
  final int classId;
  final double xc;
  final double yc;
  final double w;
  final double h;

  const GtBox(this.classId, this.xc, this.yc, this.w, this.h);
}

/// Resolves a YOLO `.txt` label path for an image (same folder or `images`→`labels`).
String? resolveYoloLabelPath(String imagePath) {
  try {
    final f = File(imagePath);
    final stem = f.path.split(RegExp(r'[/\\]')).last;
    final dot = stem.lastIndexOf('.');
    final base = dot >= 0 ? stem.substring(0, dot) : stem;
    final same = File('${f.parent.path}${Platform.pathSeparator}$base.txt');
    if (same.existsSync()) return same.path;

    final norm = imagePath.replaceAll('\\', '/');
    final parts = norm.split('/');
    final imagesIdx = parts.indexWhere((s) => s.toLowerCase() == 'images');
    if (imagesIdx >= 0) {
      final copy = List<String>.from(parts);
      copy[imagesIdx] = 'labels';
      copy[copy.length - 1] = '$base.txt';
      final candidate = copy.join(Platform.pathSeparator);
      if (File(candidate).existsSync()) return candidate;
    }
  } catch (_) {}
  return null;
}

/// Loads YOLO label lines from disk.
Future<List<GtBox>> loadYoloLabelFile(String? path) async {
  if (path == null) return [];
  final f = File(path);
  if (!await f.exists()) return [];
  final lines = await f.readAsLines();
  final out = <GtBox>[];
  for (final line in lines) {
    final parts = line.trim().split(RegExp(r'\s+'));
    if (parts.length < 5) continue;
    final cls = int.tryParse(parts[0]);
    final xc = double.tryParse(parts[1]);
    final yc = double.tryParse(parts[2]);
    final w = double.tryParse(parts[3]);
    final h = double.tryParse(parts[4]);
    if (cls == null || xc == null || yc == null || w == null || h == null) {
      continue;
    }
    out.add(GtBox(cls, xc, yc, w, h));
  }
  return out;
}

/// Carrega todos os `.txt` YOLO em [paths] (ex.: resultado completo do SAF.cache)
/// e indexa por stem do ficheiro — alinha com o nome da imagem sem extensão.
Future<Map<String, List<GtBox>>> loadYoloLabelsFromPathList(
  List<String> paths,
) async {
  final out = <String, List<GtBox>>{};
  for (final p in paths) {
    if (!p.toLowerCase().endsWith('.txt')) continue;
    final f = File(p);
    if (!await f.exists()) continue;
    final stem = baseName(p.split(RegExp(r'[/\\]')).last);
    out[stem] = await loadYoloLabelFile(p);
  }
  return out;
}

List<double> xywh2xyxyNorm(double xc, double yc, double w, double h) {
  final x1 = xc - w / 2;
  final y1 = yc - h / 2;
  final x2 = xc + w / 2;
  final y2 = yc + h / 2;
  return [x1, y1, x2, y2];
}

double iouNorm(List<double> a, List<double> b) {
  final xi1 = math.max(a[0], b[0]);
  final yi1 = math.max(a[1], b[1]);
  final xi2 = math.min(a[2], b[2]);
  final yi2 = math.min(a[3], b[3]);
  final inter = math.max(0.0, xi2 - xi1) * math.max(0.0, yi2 - yi1);
  final ar1 = (a[2] - a[0]) * (a[3] - a[1]);
  final ar2 = (b[2] - b[0]) * (b[3] - b[1]);
  final union = ar1 + ar2 - inter;
  return union > 0 ? inter / union : 0.0;
}

/// Parses prediction box map (aligned with [compute_metrics.py]).
({Object cls, double conf, List<double> xyxy}) parsePredBox(Map<String, dynamic> box) {
  double x1;
  double y1;
  double x2;
  double y2;
  if (box.containsKey('x1_norm')) {
    x1 = (box['x1_norm'] as num).toDouble();
    y1 = (box['y1_norm'] as num).toDouble();
    x2 = (box['x2_norm'] as num).toDouble();
    y2 = (box['y2_norm'] as num).toDouble();
  } else if (box['normalizedBox'] != null) {
    final nb = box['normalizedBox'] as Map;
    x1 = ((nb['left'] ?? nb['x1'] ?? 0) as num).toDouble();
    y1 = ((nb['top'] ?? nb['y1'] ?? 0) as num).toDouble();
    x2 = ((nb['right'] ?? nb['x2'] ?? 0) as num).toDouble();
    y2 = ((nb['bottom'] ?? nb['y2'] ?? 0) as num).toDouble();
  } else if (box['boundingBox'] != null) {
    final bb = box['boundingBox'] as Map;
    final iw = (box['imageWidth'] as num?)?.toDouble() ?? 1.0;
    final ih = (box['imageHeight'] as num?)?.toDouble() ?? 1.0;
    x1 = ((bb['left'] as num?)?.toDouble() ?? 0) / iw;
    y1 = ((bb['top'] as num?)?.toDouble() ?? 0) / ih;
    x2 = ((bb['right'] as num?)?.toDouble() ?? 0) / iw;
    y2 = ((bb['bottom'] as num?)?.toDouble() ?? 0) / ih;
  } else {
    return (cls: 0, conf: 0.0, xyxy: [0.0, 0.0, 0.0, 0.0]);
  }
  final conf = (box['confidence'] ?? box['conf'] ?? 0) as num? ?? 0;
  final cls = box['classIndex'] ?? box['index'] ?? box['class'] ?? 0;
  return (cls: cls, conf: conf.toDouble(), xyxy: [x1, y1, x2, y2]);
}

String baseName(String fileName) {
  final n = fileName.replaceAll('\\', '/').split('/').last;
  final dot = n.lastIndexOf('.');
  return dot >= 0 ? n.substring(0, dot) : n;
}

int normalizePredClass(
  Object pcls,
  Map<int, String> classMap,
  Map<String, int> nameToId,
) {
  if (pcls is int) return pcls;
  if (pcls is String) {
    final s = pcls.trim();
    if (RegExp(r'^-?[\d.]+$').hasMatch(s.replaceAll('-', ''))) {
      return int.tryParse(s.split('.').first) ?? -1;
    }
    return nameToId[s] ?? -1;
  }
  return -1;
}

bool clsMatchPredGt(
  Object pcls,
  int gcls,
  Map<int, String> classMap,
) {
  if (pcls is String) {
    return classMap[gcls] == pcls || '$gcls' == pcls.trim();
  }
  final pi = pcls is int ? pcls : int.tryParse('$pcls') ?? -999;
  return pi == gcls;
}

double vocAp11(List<double> recalls, List<double> precisions) {
  if (recalls.isEmpty) return 0.0;
  var ap = 0.0;
  for (var t = 0; t <= 10; t++) {
    final thr = t / 10.0;
    var pMax = 0.0;
    for (var i = 0; i < recalls.length; i++) {
      if (recalls[i] >= thr) {
        pMax = math.max(pMax, precisions[i]);
      }
    }
    ap += pMax / 11.0;
  }
  return ap;
}

/// Default class names aligned with [compute_metrics.py] / EPI dataset.
Map<int, String> defaultEpiClassMap() => {
      0: 'Hardhat',
      1: 'Mask',
      2: 'NO-Hardhat',
      3: 'NO-Mask',
      4: 'NO-Safety Vest',
      5: 'Person',
      6: 'Safety Cone',
      7: 'Safety Vest',
      8: 'machinery',
      9: 'vehicle',
    };

typedef _GtEntry = ({int cls, List<double> xy});

List<_GtEntry> _buildGtXyxyList(List<GtBox> raw) {
  return [
    for (final g in raw)
      (
        cls: g.classId,
        xy: xywh2xyxyNorm(g.xc, g.yc, g.w, g.h),
      ),
  ];
}

Map<String, dynamic> _evaluateClassVocFull(
  int classC,
  List<Map<String, dynamic>> results,
  Map<String, List<_GtEntry>> gtXyxyByBase,
  double confThreshold,
  double iouThreshold,
  int nGt,
  Map<int, String> classMap,
  Map<String, int> nameToId,
) {
  final predsC = <({double conf, String base, List<double> xy})>[];
  for (final item in results) {
    final fileName = item['file']?.toString() ?? item['path']?.toString() ?? '';
    if (fileName.isEmpty) continue;
    final bname = baseName(fileName);
    final rawBoxes = item['boxes'];
    if (rawBoxes is! List) continue;
    for (final b in rawBoxes) {
      if (b is! Map) continue;
      final parsed = parsePredBox(Map<String, dynamic>.from(b));
      if (parsed.conf < confThreshold) continue;
      if (normalizePredClass(parsed.cls, classMap, nameToId) != classC) continue;
      predsC.add((conf: parsed.conf, base: bname, xy: parsed.xyxy));
    }
  }
  predsC.sort((a, b) => b.conf.compareTo(a.conf));

  final matchedIdx = <String, Set<int>>{};
  var tpC = 0;
  var fpC = 0;
  final precisions = <double>[];
  final recalls = <double>[];

  for (final p in predsC) {
    final gtList = gtXyxyByBase[p.base] ?? const <_GtEntry>[];
    var bestJ = -1;
    var bestIou = 0.0;
    for (var j = 0; j < gtList.length; j++) {
      if (gtList[j].cls != classC) continue;
      if (matchedIdx[p.base]?.contains(j) ?? false) continue;
      final iou = iouNorm(p.xy, gtList[j].xy);
      if (iou > bestIou && iou >= iouThreshold) {
        bestIou = iou;
        bestJ = j;
      }
    }
    if (bestJ >= 0) {
      matchedIdx.putIfAbsent(p.base, () => <int>{}).add(bestJ);
      tpC++;
    } else {
      fpC++;
    }
    final den = tpC + fpC;
    final prec = den > 0 ? tpC / den : 0.0;
    final rec = nGt > 0 ? tpC / nGt : 0.0;
    precisions.add(prec);
    recalls.add(rec);
  }

  final apC = (nGt > 0 && predsC.isNotEmpty)
      ? vocAp11(recalls, precisions)
      : (nGt > 0 ? 0.0 : 0.0);
  final fnC = nGt - tpC;
  final precC = (tpC + fpC) > 0 ? tpC / (tpC + fpC) : 0.0;
  final recC = (tpC + fnC) > 0 ? tpC / (tpC + fnC) : 0.0;
  final accC = (tpC + fpC + fnC) > 0 ? tpC / (tpC + fpC + fnC) : 0.0;

  return {
    'precision': precC,
    'recall': recC,
    'accuracy': accC,
    'ap': apC,
    'tp': tpC,
    'fp': fpC,
    'fn': fnC,
  };
}

/// Computes detection metrics from batch inference [results] and [gtAll] (stem → YOLO boxes).
Map<String, dynamic> computeDetectionMetrics(
  List<Map<String, dynamic>> results,
  Map<String, List<GtBox>> gtAll, {
  double iouThreshold = 0.5,
  double confThreshold = 0.001,
  Map<int, String>? classMap,
}) {
  final cm = classMap ?? defaultEpiClassMap();
  final nameToId = <String, int>{for (final e in cm.entries) e.value: e.key};

  final gtXyxyByBase = <String, List<_GtEntry>>{};
  for (final e in gtAll.entries) {
    gtXyxyByBase[e.key] = _buildGtXyxyList(e.value);
  }

  final allPreds =
      <({double conf, String base, Object pcls, List<double> xy})>[];
  var totalImages = 0;

  for (final item in results) {
    final fn = item['file']?.toString() ?? item['path']?.toString() ?? '';
    if (fn.isEmpty) continue;
    final bname = baseName(fn);
    totalImages++;
    final rawBoxes = item['boxes'];
    if (rawBoxes is! List) continue;
    for (final b in rawBoxes) {
      if (b is! Map) continue;
      final parsed = parsePredBox(Map<String, dynamic>.from(b));
      if (parsed.conf >= confThreshold) {
        allPreds.add((
          conf: parsed.conf,
          base: bname,
          pcls: parsed.cls,
          xy: parsed.xyxy,
        ));
      }
    }
  }

  allPreds.sort((a, b) => b.conf.compareTo(a.conf));

  final gtMatchedGlobal = <String, List<bool>>{};
  var tpTotal = 0;
  var fpTotal = 0;
  final perClassFp = <int, int>{};

  for (final p in allPreds) {
    final gtList = gtXyxyByBase[p.base] ?? const <_GtEntry>[];
    gtMatchedGlobal.putIfAbsent(
      p.base,
      () => List<bool>.filled(gtList.length, false),
    );
    if (gtMatchedGlobal[p.base]!.length != gtList.length) {
      gtMatchedGlobal[p.base] = List<bool>.filled(gtList.length, false);
    }
    final gm = gtMatchedGlobal[p.base]!;

    var bestIou = 0.0;
    var bestJ = -1;

    for (var j = 0; j < gtList.length; j++) {
      if (gm[j]) continue;
      final gcls = gtList[j].cls;
      final gxy = gtList[j].xy;
      if (!clsMatchPredGt(p.pcls, gcls, cm)) continue;
      final iou = iouNorm(p.xy, gxy);
      if (iou > bestIou && iou >= iouThreshold) {
        bestIou = iou;
        bestJ = j;
      }
    }

    if (bestJ >= 0) {
      gm[bestJ] = true;
      tpTotal++;
    } else {
      fpTotal++;
      final k = normalizePredClass(p.pcls, cm, nameToId);
      perClassFp[k] = (perClassFp[k] ?? 0) + 1;
    }
  }

  var fnTotal = 0;

  for (final item in results) {
    final fn = item['file']?.toString() ?? item['path']?.toString() ?? '';
    if (fn.isEmpty) continue;
    final bname = baseName(fn);
    final gtList = gtXyxyByBase[bname] ?? const <_GtEntry>[];
    var gm = gtMatchedGlobal[bname];
    gm ??= List<bool>.filled(gtList.length, false);
    if (gm.length != gtList.length) {
      gm = List<bool>.filled(gtList.length, false);
    }
    for (var j = 0; j < gtList.length; j++) {
      if (!gm[j]) {
        fnTotal++;
      }
    }
  }

  final precision =
      (tpTotal + fpTotal) > 0 ? tpTotal / (tpTotal + fpTotal) : 0.0;
  final recall =
      (tpTotal + fnTotal) > 0 ? tpTotal / (tpTotal + fnTotal) : 0.0;
  final f1 = (precision + recall) > 0
      ? 2 * precision * recall / (precision + recall)
      : 0.0;
  final denomAll = tpTotal + fpTotal + fnTotal;
  final accuracy = denomAll > 0 ? tpTotal / denomAll : 0.0;

  final gtClassCounts = <int, int>{};
  for (final e in gtAll.entries) {
    for (final g in e.value) {
      gtClassCounts[g.classId] = (gtClassCounts[g.classId] ?? 0) + 1;
    }
  }

  final classesWithGt = gtClassCounts.keys.where((c) => gtClassCounts[c]! > 0).toList()
    ..sort();

  final apList = <double>[];
  final perClassVoc = <String, Map<String, dynamic>>{};

  for (final c in classesWithGt) {
    final nGt = gtClassCounts[c] ?? 0;
    final voc = _evaluateClassVocFull(
      c,
      results,
      gtXyxyByBase,
      confThreshold,
      iouThreshold,
      nGt,
      cm,
      nameToId,
    );
    perClassVoc['$c'] = voc;
    apList.add((voc['ap'] as num).toDouble());
  }

  final map50 =
      apList.isEmpty ? 0.0 : apList.reduce((a, b) => a + b) / apList.length;

  if (perClassFp.containsKey(-1) && perClassFp[-1]! > 0) {
    perClassVoc['-1'] = {
      'precision': 0.0,
      'recall': 0.0,
      'accuracy': 0.0,
      'ap': 0.0,
      'tp': 0,
      'fp': perClassFp[-1],
      'fn': 0,
      'note': 'nome de classe não mapeado',
    };
  }

  return {
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'accuracy': accuracy,
    'mAP@0.5': map50,
    'tp': tpTotal,
    'fp': fpTotal,
    'fn': fnTotal,
    'total_images': totalImages,
    'gt_labels_loaded': gtAll.length,
    'per_class': perClassVoc,
  };
}
