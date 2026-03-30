#!/usr/bin/env python3
"""
Calcula métricas de detecção (P/R/F1/accuracy globais + mAP@0.5 estilo VOC)
comparando o JSON de inferência do app Flutter com labels YOLO (GT).

TP = IoU >= limiar e classe correta (índice ou nome via --classes).
Globais: ordenação por confiança em todo o lote (estilo COCO). Por classe: AP
com curva PR em avaliação independente (preds da classe vs GT da classe).

IMPORTANTE: Os nomes dos arquivos na inferência devem corresponder às labels.
Ex: imagem "youtube-836_jpg.rf.xxx.jpg" -> label "youtube-836_jpg.rf.xxx.txt"

Uso:
    python compute_metrics.py ^
        --inference "D:\Docs\SafeAI\construction site\660-11f1-9301-c749b0e9fa98.json" ^
        --labels "D:\Docs\SafeAI\construction site\dataset_construction_site\css-data\valid\labels" ^
        --classes "0:Hardhat,1:Mask,2:NO-Hardhat,3:NO-Mask,4:NO-Safety Vest,5:Person,6:Safety Cone,7:Safety Vest,8:machinery,9:vehicle"
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_inference_json(path: str) -> list[dict]:
    """Carrega o JSON de inferência (formato do app Flutter batch)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data if isinstance(data, list) else [data]


def load_yolo_labels(labels_dir: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """
    Carrega labels YOLO (class_id x_center y_center width height, normalizado).
    Retorna dict: base_name -> [(class_id, xc, yc, w, h), ...]
    """
    gt = {}
    for lbl_path in labels_dir.glob("*.txt"):
        base = lbl_path.stem
        boxes = []
        with open(lbl_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append((cls_id, xc, yc, w, h))
        gt[base] = boxes
    return gt


def xywh2xyxy_norm(xc: float, yc: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Converte YOLO (center, size) normalizado para xyxy normalizado."""
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return (x1, y1, x2, y2)


def parse_pred_box(box: dict) -> tuple[int | str, float, tuple[float, float, float, float]]:
    """
    Extrai (class, confidence, (x1,y1,x2,y2) norm) do box de predição.
    Aceita: x1_norm/y1_norm/x2_norm/y2_norm ou left/top/right/bottom em normalizedBox.
    Class pode ser int (index) ou str (nome).
    """
    # Coordenadas normalizadas
    if "x1_norm" in box:
        x1 = float(box["x1_norm"])
        y1 = float(box["y1_norm"])
        x2 = float(box["x2_norm"])
        y2 = float(box["y2_norm"])
    elif "normalizedBox" in box:
        nb = box["normalizedBox"]
        x1 = float(nb.get("left", nb.get("x1", 0)))
        y1 = float(nb.get("top", nb.get("y1", 0)))
        x2 = float(nb.get("right", nb.get("x2", 0)))
        y2 = float(nb.get("bottom", nb.get("y2", 0)))
    elif "boundingBox" in box:
        bb = box["boundingBox"]
        # Pixel coords - precisamos de image size para normalizar; usa 1 se não tiver
        w = float(box.get("imageWidth", 1))
        h = float(box.get("imageHeight", 1))
        x1 = float(bb.get("left", 0)) / w
        y1 = float(bb.get("top", 0)) / h
        x2 = float(bb.get("right", 0)) / w
        y2 = float(bb.get("bottom", 0)) / h
    else:
        return (0, 0.0, (0.0, 0.0, 0.0, 0.0))

    conf = float(box.get("confidence", box.get("conf", 0)))
    cls = box.get("classIndex", box.get("index", box.get("class", 0)))
    if isinstance(cls, str) and cls.replace("-", "").replace(".", "").isdigit():
        cls = int(float(cls))
    return (cls, conf, (x1, y1, x2, y2))


def iou_norm(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """IoU entre duas boxes normalizadas (x1,y1,x2,y2)."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def get_base_name(file_name: str) -> str:
    """Extrai nome base sem extensão."""
    return Path(file_name).stem


def normalize_pred_class(
    pcls: int | str,
    class_map: dict[int, str],
    name_to_id: dict[str, int],
) -> int:
    """Índice inteiro da classe prevista (para match e FP); -1 se nome desconhecido."""
    if isinstance(pcls, int):
        return int(pcls)
    if isinstance(pcls, str):
        s = pcls.strip()
        if s.replace("-", "").replace(".", "").isdigit():
            return int(float(s))
        return name_to_id.get(s, -1)
    try:
        return int(pcls)
    except (ValueError, TypeError):
        return -1


def cls_match_pred_gt(
    pcls: int | str,
    gcls: int,
    class_map: dict[int, str],
) -> bool:
    """Predição e GT são a mesma classe (índice ou nome)."""
    if isinstance(pcls, str):
        return class_map.get(gcls) == pcls or str(gcls) == pcls.strip()
    try:
        return int(pcls) == gcls
    except (ValueError, TypeError):
        return class_map.get(gcls) == str(pcls)


def voc_ap11(recalls: list[float], precisions: list[float]) -> float:
    """Average Precision (interpolação 11 pontos, estilo VOC)."""
    if not recalls:
        return 0.0
    ap = 0.0
    for t in range(11):
        thr = t / 10.0
        p_max = 0.0
        for r, p in zip(recalls, precisions):
            if r >= thr:
                p_max = max(p_max, p)
        ap += p_max / 11.0
    return ap


def compute_metrics(
    inference_path: str,
    labels_dir: str,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.001,
    class_map: dict[int, str] | None = None,
    name_to_id: dict[str, int] | None = None,
    debug: bool = False,
) -> dict:
    """
    Métricas no estilo PASCAL VOC / COCO:
    - P/R/F1/accuracy globais: uma ordenação global por confiança; TP só com classe correta e IoU.
    - Por classe: P/R e AP@IoU a partir de avaliação independente (preds da classe vs GT da classe).
    - mAP@0.5: média do AP só nas classes com pelo menos um GT no conjunto avaliado.
    """
    if class_map is None:
        class_map = {}
    if name_to_id is None:
        name_to_id = {v: k for k, v in class_map.items()}
    data = load_inference_json(inference_path)
    gt_all = load_yolo_labels(Path(labels_dir))

    def build_gt_xyxy(base: str) -> list[tuple[int, tuple[float, float, float, float]]]:
        return [
            (cls_id, xywh2xyxy_norm(xc, yc, w, h))
            for cls_id, xc, yc, w, h in gt_all.get(base, [])
        ]

    # --- 1) Lista global de predições (conf decrescente = ordem COCO-style) ---
    all_preds: list[tuple[float, str, int | str, tuple[float, float, float, float]]] = []
    matched_images = 0
    total_images = 0
    debug_count = 0
    max_debug = 3

    for item in data:
        file_name = item.get("file", item.get("path", ""))
        if not file_name:
            continue
        base = get_base_name(file_name)
        pred_boxes_raw = item.get("boxes", [])
        for b in pred_boxes_raw:
            pcls, conf, xyxy = parse_pred_box(b)
            if conf >= conf_threshold:
                all_preds.append((conf, base, pcls, xyxy))
        total_images += 1
        gt_xy = build_gt_xyxy(base)
        if not gt_xy and not pred_boxes_raw:
            matched_images += 1

    all_preds.sort(key=lambda x: -x[0])

    # --- 2) Match global: TP só se classe prevista == GT e IoU >= limiar ---
    gt_matched_global: dict[str, list[bool]] = {}
    tp_total = fp_total = fn_total = 0
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)

    for conf, base, pcls, pxyxy in all_preds:
        if base not in gt_matched_global:
            gt_matched_global[base] = [False] * len(build_gt_xyxy(base))
        gt_xyxy = build_gt_xyxy(base)
        if len(gt_matched_global[base]) != len(gt_xyxy):
            gt_matched_global[base] = [False] * len(gt_xyxy)

        gm = gt_matched_global[base]
        best_iou = 0.0
        best_j = -1
        best_iou_any = 0.0

        for j, (gcls, gxyxy) in enumerate(gt_xyxy):
            if gm[j]:
                continue
            iou = iou_norm(pxyxy, gxyxy)
            if iou > best_iou_any:
                best_iou_any = iou
            if not cls_match_pred_gt(pcls, gcls, class_map):
                continue
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_j = j

        if debug and debug_count < max_debug and best_j < 0 and all_preds:
            debug_count += 1
            print(f"\n[DEBUG] {base} | pred cls={pcls} conf={conf:.2f} box={[f'{x:.3f}' for x in pxyxy]}")
            print(
                f"        gt boxes: {len(gt_xyxy)} | best_iou_same_class={best_iou:.3f} "
                f"best_iou_any={best_iou_any:.3f} threshold={iou_threshold}"
            )
            for j, (gcls, gxyxy) in enumerate(gt_xyxy):
                iou = iou_norm(pxyxy, gxyxy)
                cname = class_map.get(gcls, "?")
                same = (
                    "OK"
                    if cls_match_pred_gt(pcls, gcls, class_map)
                    else "cls_diff"
                )
                if iou > 0.05:
                    print(f"          gt[{j}] cls={gcls}({cname}) iou={iou:.3f} {same}")

        if best_j >= 0:
            gm[best_j] = True
            tp_total += 1
            matched_cls = gt_xyxy[best_j][0]
            per_class_tp[matched_cls] += 1
        else:
            fp_total += 1
            k = normalize_pred_class(pcls, class_map, name_to_id)
            per_class_fp[k] += 1

    for item in data:
        file_name = item.get("file", item.get("path", ""))
        if not file_name:
            continue
        base = get_base_name(file_name)
        gt_xyxy = build_gt_xyxy(base)
        gm = gt_matched_global.get(base, [False] * len(gt_xyxy))
        if len(gm) != len(gt_xyxy):
            gm = [False] * len(gt_xyxy)
        for j, m in enumerate(gm):
            if not m:
                fn_total += 1
                gcls = gt_xyxy[j][0]
                per_class_fn[gcls] += 1
        if gt_xyxy or any(
            conf >= conf_threshold
            for conf, b, _, _ in all_preds
            if b == base
        ):
            matched_images += 1

    # Métricas globais (micro)
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    denom_all = tp_total + fp_total + fn_total
    accuracy = tp_total / denom_all if denom_all > 0 else 0.0

    # --- 3) Por classe (VOC): AP@IoU, P/R com match só entre pred e GT da mesma classe ---
    gt_class_counts: dict[int, int] = defaultdict(int)
    for item in data:
        fn = item.get("file", item.get("path", ""))
        if not fn:
            continue
        base = get_base_name(fn)
        for cls_id, *_ in gt_all.get(base, []):
            gt_class_counts[cls_id] += 1

    def evaluate_class_voc(class_c: int) -> tuple[int, int, int, float]:
        n_gt = gt_class_counts.get(class_c, 0)
        preds_c: list[tuple[float, str, tuple[float, float, float, float]]] = []
        for item in data:
            file_name = item.get("file", item.get("path", ""))
            if not file_name:
                continue
            base = get_base_name(file_name)
            for b in item.get("boxes", []):
                pcls, conf, xyxy = parse_pred_box(b)
                if conf < conf_threshold:
                    continue
                if normalize_pred_class(pcls, class_map, name_to_id) != class_c:
                    continue
                preds_c.append((conf, base, xyxy))
        preds_c.sort(key=lambda x: -x[0])

        matched_idx: dict[str, set[int]] = defaultdict(set)
        tp_c = fp_c = 0
        precisions: list[float] = []
        recalls: list[float] = []

        for conf, base, pxyxy in preds_c:
            gt_xyxy = build_gt_xyxy(base)
            best_j = -1
            best_iou = 0.0
            for j, (gcls, gxyxy) in enumerate(gt_xyxy):
                if gcls != class_c:
                    continue
                if j in matched_idx[base]:
                    continue
                iou = iou_norm(pxyxy, gxyxy)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                matched_idx[base].add(best_j)
                tp_c += 1
            else:
                fp_c += 1
            den = tp_c + fp_c
            prec = tp_c / den if den > 0 else 0.0
            rec = tp_c / n_gt if n_gt > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

        if n_gt > 0 and preds_c:
            ap_c = voc_ap11(recalls, precisions)
        elif n_gt > 0:
            ap_c = 0.0
        else:
            ap_c = 0.0
        fn_c = n_gt - tp_c
        return tp_c, fp_c, fn_c, ap_c

    classes_with_gt = {c for c, n in gt_class_counts.items() if n > 0}
    per_class_voc: dict[int, dict] = {}
    ap_list: list[float] = []

    for c in sorted(classes_with_gt):
        tp_c, fp_c, fn_c, ap_c = evaluate_class_voc(int(c))
        prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        acc_c = tp_c / (tp_c + fp_c + fn_c) if (tp_c + fp_c + fn_c) > 0 else 0.0
        per_class_voc[int(c)] = {
            "precision": prec_c,
            "recall": rec_c,
            "accuracy": acc_c,
            "ap": ap_c,
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
        }
        ap_list.append(ap_c)

    map50 = sum(ap_list) / len(ap_list) if ap_list else 0.0

    # FP com classe desconhecida (-1), se houver
    if per_class_fp.get(-1, 0) > 0:
        per_class_voc[-1] = {
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "ap": 0.0,
            "tp": 0,
            "fp": per_class_fp[-1],
            "fn": 0,
            "note": "nome de classe não mapeado em --classes",
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "mAP@0.5": map50,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "images_matched": matched_images,
        "total_images": total_images,
        "gt_labels_loaded": len(gt_all),
        "per_class": per_class_voc,
    }


def main():
    parser = argparse.ArgumentParser(description="Calcula métricas de detecção")
    parser.add_argument(
        "--inference",
        "-i",
        required=True,
        help="Caminho do JSON de inferência",
    )
    parser.add_argument(
        "--labels",
        "-l",
        required=True,
        help="Pasta com as labels YOLO (valid/labels)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold para match (default: 0.5)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold mínimo (default: 0.001)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Salvar métricas em JSON",
    )
    parser.add_argument(
        "--classes",
        "-c",
        help="Mapeamento classe (ex: '0:Hardhat,1:Mask,7:Safety Vest') para match nome<->id",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mostra amostras de comparação pred vs gt (primeiras imagens)",
    )
    args = parser.parse_args()

    # Mapeamento padrão para dataset EPI/construction (ppe_data.yaml)
    # IMPORTANTE: ordem deve coincidir com o modelo TFLite e as labels YOLO
    class_map = {
        0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask",
        4: "NO-Safety Vest", 5: "Person", 6: "Safety Cone",
        7: "Safety Vest", 8: "machinery", 9: "vehicle",
    }
    if args.classes:
        for part in args.classes.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                class_map[int(k.strip())] = v.strip()
    name_to_id = {v: k for k, v in class_map.items()}

    metrics = compute_metrics(
        args.inference,
        args.labels,
        iou_threshold=args.iou,
        conf_threshold=args.conf,
        class_map=class_map,
        name_to_id=name_to_id,
        debug=args.debug,
    )

    print("\n=== Métricas de Detecção ===\n")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-Score:   {metrics['f1']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}  (TP / (TP+FP+FN))")
    print(f"mAP@0.5:    {metrics['mAP@0.5']:.4f}  (média AP VOC por classe com GT)")
    print(f"\nTP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}")
    print(f"Imagens: {metrics['total_images']} | Labels carregadas: {metrics['gt_labels_loaded']}")

    # Aviso se os arquivos não batem (nomes diferentes)
    if metrics["tp"] == 0 and metrics["fp"] > 0:
        print("\n⚠️  TP=0 (nenhum true positive). Possíveis causas:")
        print("   1. Mismatch de classes: modelo treinado com ordem diferente do dataset")
        print("   2. Coordenadas/IoU: boxes não se sobrepõem (IoU < threshold)")
        print("   3. Nomes de arquivos: inferência em imagens diferentes das labels")
        print("\n   Tente: --iou 0.3 (ou 0.2) e --debug para mais detalhes")

    if metrics["per_class"]:
        print("\n--- Por classe (match só com GT da mesma classe; AP = VOC 11-pt) ---")
        for cls, m in sorted(metrics["per_class"].items()):
            if "note" in m:
                print(f"  Classe {cls}: FP={m['fp']} ({m['note']})")
                continue
            prec = m["precision"]
            rec = m["recall"]
            acc = m["accuracy"]
            ap = m.get("ap", 0.0)
            print(
                f"  Classe {cls}: P={prec:.3f} R={rec:.3f} Acc={acc:.3f} AP={ap:.3f} "
                f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})"
            )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nMétricas salvas em: {args.output}")


if __name__ == "__main__":
    main()
