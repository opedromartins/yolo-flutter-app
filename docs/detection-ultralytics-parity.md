# Alinhamento das deteções com Ultralytics / avaliação COCO

Este documento descreve as alterações feitas no projeto para **aproximar ao máximo** os resultados das deteções no Android/Flutter aos obtidos com `YOLO(...).predict()` em Python (mesmo modelo `.tflite`, métricas COCO com python em PC).

Referência mental: script Python com `imgsz=640`, `conf=0.001`, `iou` NMS `0.7`, `max_det=300`, letterbox e `pycocotools`.

---

## 1. Pré-processamento: letterbox (Android)

**Problema:** O pipeline antigo usava `ResizeOp` bilinear para esticar a imagem ao tamanho de entrada do tensor, **sem** manter a proporção. O Ultralytics usa **letterbox**: escala mantendo aspect ratio, preenche com cinza **(114, 114, 114)** e centra a imagem no tensor.

**Alteração:** Em `ObjectDetector.kt` foi implementado letterbox manual:

- Cálculo do fator `gain = min(inW/w, inH/h)` (como no Ultralytics).
- Redimensionamento da imagem para `newW × newH` e desenho num canvas `inW × inH` com padding.
- Normalização **RGB / 255** para o `ByteBuffer` de entrada do TFLite (equivalente ao `NormalizeOp(0, 255)`).

**Ficheiro:** `android/src/main/kotlin/com/ultralytics/yolo/ObjectDetector.kt`

**Mapeamento de caixas:** As coordenadas devolvidas pelo JNI estão normalizadas ao **tensor letterboxed**. Foram adicionadas funções que convertem de volta para coordenadas na imagem original:

- `(x_pixel_no_tensor - padX) / gain` (e o mesmo para `y`, e para cantos inferiores / largura-altura).
- Ramo **E2E** (`[N, 6]`): mesmo princípio para `xyxy` normalizados ou em pixels do tensor.

A rotação da câmara (retrato: traseira 270° CCW em 3 passos, frontal 90° CCW) mantém-se equivalente ao antigo `Rot90Op`, mas **antes** do letterbox, sobre `Bitmap`, para depois letterbox único.

---

## 2. Pós-processamento JNI: NMS por classe (C++)

**Problema:** O NMS em `native-lib.cpp` era **agnóstico à classe**: caixas de classes diferentes com IoU alto podiam suprimir-se mutuamente, ao contrário do Ultralytics (NMS **por classe**).

**Alteração:**

1. Agrupar propostas por `class_index`.
2. Para cada classe: ordenar por confiança (desc.) e aplicar NMS **só dentro dessa classe**.
3. Juntar todas as caixas aprovadas, ordenar globalmente por confiança (desc.).
4. Truncar a `num_items_threshold` (equivalente a `max_det`).

**Ficheiro:** `android/src/main/cpp/native-lib.cpp`

---

## 3. Limiares e limites por defeito (métricas COCO)

Objetivo: alinhar com avaliação típica COCO no Ultralytics (conf muito baixa para não cortar candidatos antes do NMS final; IoU NMS 0.7; muitas deteções por imagem).

| Parâmetro            | Valor por defeito (após alterações) | Nota |
|----------------------|-------------------------------------|------|
| Confiança mínima     | `0.001`                             | Igual ordem de grandeza ao `conf=0.001` do `predict` para métricas |
| IoU (NMS)            | `0.7`                               | Alinhado ao `iou` do NMS no Ultralytics |
| Máx. deteções        | `300`                               | Próximo de `max_det=300` COCO |

**Android (Kotlin):**

- `ObjectDetector.kt`: campos privados `confidenceThreshold`, `iouThreshold`; construtor `numItemsThreshold = 300`.
- `YOLO.kt`: `numItemsThreshold` por defeito `300`.
- `YOLOView.kt`: `confidenceThreshold`, `iouThreshold`, `numItemsThreshold` iniciais coerentes; etiqueta de confiança nativa ajustada ao texto inicial.
- `YOLOPlugin.kt` / `YOLOInstanceManager.kt`: fallback ao carregar modelo sem argumentos (`numItemsThreshold` default `300`).

**Dart (Flutter):**

- `lib/widgets/yolo_controller.dart`: defaults `0.001`, `0.7`, `300`; `numItemsThreshold` aceita **1–500** (antes limitava a 100).
- `lib/yolo_view.dart`: defaults do widget alinhados.
- `lib/yolo.dart` e `lib/core/yolo_model_manager.dart`: `numItemsThreshold ?? 300` e comentário sobre confiança estilo COCO.

**Exemplo:**

- `example/lib/presentation/controllers/camera_inference_controller.dart`: default `300` para consistência na app de exemplo.

---

## 4. Testes e mocks

Os testes que fixavam valores antigos (ex.: conf `0.5`, IoU `0.45`, `numItems` `30`) foram atualizados para refletir os novos defaults e o `loadModel` com `numItemsThreshold: 300`.

**Pastas:** `test/`, `test/utils/test_helpers.dart`.

---

## 5. O que continua a diferir de rodar com python no notebook (limitações esperadas)

Mesmo com o acima, pequenas diferenças de mAP ou de caixas podem permanecer:

- **TFLite** no dispositivo (FP16, delegates GPU/NPU) vs **PyTorch** no PC (FP32): erros de arredondamento.
- **Letterbox:** possíveis diferenças de 1 pixel em `round` vs OpenCV/NumPy no Ultralytics.
- **NMS:** implementação em C++ (loop clássico) vs kernels PyTorch; deve estar muito próximo, não garantido bit-a-bit.
- **iOS:** este documento foca nas alterações Android; o código Swift não foi alterado neste trabalho.

Para uma UI mais “limpa” no dia a dia, podes subir a confiança no controlador (ex. `0.25`); os defaults favorecem **paridade com métricas**, não necessariamente o overlay mais legível.

---

## 6. Índice de ficheiros tocados (resumo)

| Área | Ficheiros |
|------|-----------|
| Letterbox + mapeamento + defaults detector | `android/.../ObjectDetector.kt` |
| NMS por classe | `android/.../cpp/native-lib.cpp` |
| Defaults YOLO / vista / plugin / instância | `YOLO.kt`, `YOLOView.kt`, `YOLOPlugin.kt`, `YOLOInstanceManager.kt` |
| Flutter: thresholds e max det | `lib/widgets/yolo_controller.dart`, `lib/yolo_view.dart`, `lib/yolo.dart`, `lib/core/yolo_model_manager.dart` |
| Exemplo | `example/.../camera_inference_controller.dart` |
| Testes | `test/**/*.dart`, `test/utils/test_helpers.dart` |

---

*Última atualização: documentação das alterações de paridade Ultralytics/COCO neste repositório.*
