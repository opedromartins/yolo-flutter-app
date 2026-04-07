#!/usr/bin/env python3
"""
Desenha bboxes nas imagens: Pred = verde, GT = azul.
Por padrão só predições. Use --labels para incluir GT (azul).

Uso:
    python draw_boxes_on_images.py ^
        -i "inferencia.json" ^
        -im "pasta_images" ^
        -o "pasta_saida" ^
        -n 20 ^
        [ -l "pasta_labels" para desenhar GT também ]
"""

import argparse
import json
import random
from pathlib import Path

# Cores GT e Pred
# GT = azul | Pred = verde
COLOR_GT_CV2 = (255, 0, 0)   # azul (BGR)
COLOR_GT_PIL = (0, 0, 255)   # azul (RGB)
COLOR_PRED_CV2 = (0, 255, 0)  # verde (BGR)
COLOR_PRED_PIL = (0, 255, 0)  # verde (RGB)

CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]


def load_inference_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_yolo_labels(labels_dir: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    gt = {}
    for lbl_path in labels_dir.glob("*.txt"):
        base = lbl_path.stem
        boxes = []
        with open(lbl_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append((cls_id, xc, yc, w, h))
        gt[base] = boxes
    return gt


def xywh_norm_to_xyxy_px(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int):
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))


def draw_boxes_on_image(
    img_path: Path,
    out_path: Path,
    pred_boxes: list[dict] | None = None,
    gt_boxes: list[tuple[int, float, float, float, float]] | None = None,
    class_names: list[str] = CLASS_NAMES,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        try:
            import cv2
            use_cv2 = True
        except ImportError:
            print("Erro: instale Pillow (pip install Pillow) ou OpenCV (pip install opencv-python)")
            return False
    else:
        use_cv2 = False

    # Carregar imagem
    if use_cv2:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        h, w = img.shape[:2]
    else:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

    def draw_rect(x1: int, y1: int, x2: int, y2: int, color: tuple, label: str, thick: int = 2):
        if use_cv2:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thick)
            draw.text((x1, y1 - 12), label, fill=color)

    # Predições (verde, espessura 2) - usar norm para compatibilidade com qualquer resolução
    if pred_boxes:
        for b in pred_boxes:
            if "x1_norm" in b:
                x1 = int(b["x1_norm"] * w)
                y1 = int(b["y1_norm"] * h)
                x2 = int(b["x2_norm"] * w)
                y2 = int(b["y2_norm"] * h)
            elif "x1" in b:
                x1, y1 = int(b["x1"]), int(b["y1"])
                x2, y2 = int(b["x2"]), int(b["y2"])
            else:
                continue
            cls_name = b.get("className", b.get("class", "?"))
            conf = b.get("confidence", 0)
            label = f"{cls_name} {conf:.2f}"
            color = COLOR_PRED_CV2 if use_cv2 else COLOR_PRED_PIL
            draw_rect(x1, y1, x2, y2, color, label, 2)

    # Ground truth (azul)
    if gt_boxes:
        for cls_id, xc, yc, bw, bh in gt_boxes:
            x1, y1, x2, y2 = xywh_norm_to_xyxy_px(xc, yc, bw, bh, w, h)
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            label = f"GT:{cls_name}"
            color = COLOR_GT_CV2 if use_cv2 else COLOR_GT_PIL
            draw_rect(x1, y1, x2, y2, color, label, 2)

    # Legenda (canto superior esquerdo)
    legend_y = 8
    if use_cv2:
        if pred_boxes:
            cv2.rectangle(img, (8, legend_y - 4), (28, legend_y + 10), COLOR_PRED_CV2, 2)
            cv2.putText(img, "Pred", (32, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED_CV2, 1)
            legend_y += 24
        if gt_boxes:
            cv2.rectangle(img, (8, legend_y - 4), (28, legend_y + 10), COLOR_GT_CV2, 2)
            cv2.putText(img, "GT", (32, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GT_CV2, 1)
    else:
        draw = ImageDraw.Draw(img)
        if pred_boxes:
            draw.rectangle([8, legend_y, 28, legend_y + 14], outline=COLOR_PRED_PIL, width=2)
            draw.text((32, legend_y), "Pred", fill=COLOR_PRED_PIL)
            legend_y += 24
        if gt_boxes:
            draw.rectangle([8, legend_y, 28, legend_y + 14], outline=COLOR_GT_PIL, width=2)
            draw.text((32, legend_y), "GT", fill=COLOR_GT_PIL)

    # Salvar
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if use_cv2:
        cv2.imwrite(str(out_path), img)
    else:
        img.save(out_path, quality=95)

    return True


def main():
    parser = argparse.ArgumentParser(description="Desenha bbox nas imagens")
    parser.add_argument("--inference", "-i", required=True, help="JSON de inferência")
    parser.add_argument("--images", "-im", required=True, help="Pasta com as imagens (valid/images)")
    parser.add_argument("--output", "-o", required=True, help="Pasta de saída")
    parser.add_argument("--labels", "-l", help="Opcional: pasta com labels YOLO para desenhar GT (vermelho) além da inferência")
    parser.add_argument("--max", "-n", type=int, default=20, help="Máx. imagens a salvar (default: 20)")
    parser.add_argument("--random", action="store_true", help="Amostrar aleatoriamente (senão, primeiras N)")
    args = parser.parse_args()

    data = load_inference_json(args.inference)
    images_dir = Path(args.images)
    out_dir = Path(args.output)
    labels_dir = Path(args.labels) if args.labels else None

    gt_all = load_yolo_labels(labels_dir) if labels_dir and labels_dir.exists() else {}

    # Selecionar imagens
    indices = list(range(len(data)))
    if args.random:
        random.shuffle(indices)
    indices = indices[: args.max]

    saved = 0
    for i in indices:
        item = data[i]
        fname = item.get("file", item.get("path", ""))
        if not fname:
            continue
        fname = Path(fname).name
        img_path = images_dir / fname
        if not img_path.exists():
            # Tentar sem extensão ou com variação
            base = Path(fname).stem
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                cand = images_dir / (base + ext)
                if cand.exists():
                    img_path = cand
                    break
            if not img_path.exists():
                print(f"  Imagem não encontrada: {img_path}")
                continue

        base = Path(fname).stem
        pred_boxes = item.get("boxes", [])
        gt_boxes = gt_all.get(base, []) if gt_all else []

        out_path = out_dir / f"viz_{fname}"
        if draw_boxes_on_image(img_path, out_path, pred_boxes=pred_boxes, gt_boxes=gt_boxes):
            saved += 1
            print(f"  Salvo: {out_path.name} (pred:{len(pred_boxes)} gt:{len(gt_boxes)})")

    print(f"\nTotal: {saved} imagens salvas em {out_dir}")


if __name__ == "__main__":
    main()
