// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:ultralytics_yolo/models/yolo_task.dart';

enum ModelType {
  detect('exp_yolo26m_epi_negative_float32', YOLOTask.detect),
  detectFloat16('exp_yolo26m_epi_negative_float16', YOLOTask.detect),
  yolo11nFloat16('yolo11n_float16', YOLOTask.detect),
  segment('yolo11n-seg', YOLOTask.segment),
  classify('yolo11n-cls', YOLOTask.classify),
  pose('yolo11n-pose', YOLOTask.pose),
  obb('yolo11n-obb', YOLOTask.obb),
  customFloat32('model_float32', YOLOTask.detect),                   
  customFloat16('model_float16', YOLOTask.detect),                   
  customInt8('model_int8', YOLOTask.detect);  

  final String modelName; 

  final YOLOTask task;

  const ModelType(this.modelName, this.task);
}

enum SliderType { none, numItems, confidence, iou }
