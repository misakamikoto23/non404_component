"""Create isolated annotation exercises from prepared TRAIN diagrams only."""
import json
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/prepared/manifest.json").read_text(encoding="utf-8"))
    samples = sorted((s for s in manifest["samples"] if s["split"] == "train"),
                     key=lambda s: (len(s["labels"]), s["case"]))[:6]
    target = root / "data/annotation_practice"
    target.mkdir(exist_ok=True)
    classes = manifest["classes"]
    for mode in ("blank", "review"):
        folder = target / mode
        folder.mkdir(exist_ok=True)
        (folder / "classes.txt").write_text("\n".join(classes)+"\n", encoding="utf-8")
        for sample in samples:
            case = sample["case"]
            source = root / "data/prepared/detection/images/train" / (case + ".png")
            image = folder / source.name
            if not image.exists():
                shutil.copy2(source, image)
            if mode == "review":
                from PIL import Image
                with Image.open(source) as im:
                    w, h = im.size
                shapes = [{"label": row["type"], "points": [row["bbox"][:2], row["bbox"][2:]],
                           "shape_type": "rectangle", "group_id": None, "flags": {},
                           "description": json.dumps({"source_id": row["source_id"], "Name": row["Name"], "value": row["value"]}, ensure_ascii=False)}
                          for row in sample["labels"]]
                annotation = {"version": "4.0.0", "flags": {}, "shapes": shapes,
                              "imagePath": image.name, "imageData": None, "imageHeight": h, "imageWidth": w}
                path = folder / (case + ".json")
                if not path.exists():
                    path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "selection.json").write_text(json.dumps([{"case": s["case"], "components": len(s["labels"]), "source": s["source"]} for s in samples], ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "README.md").write_text("# 标注练习\n\nblank/ 是六张训练图的副本，没有框；review/ 是同样图片与 X-AnyLabeling 格式的已有矩形标注。打开文件夹即可练习。classes.txt 固定类别顺序。\n\nreview 标签来自官方数据，仅转换了坐标和格式，不代表已人工复核。不得把练习副本作为新增独立样本混入训练/验证；原始图片和正式划分保持在 data/prepared。\n\n练习：打开 blank → 框选所有目标并填写类别 → 保存 JSON → 与 review 对比。导出 YOLO 水平矩形检测格式，使用 classes.txt 的顺序。不要将 R1、10k 等文字设为元件类别。\n", encoding="utf-8")
    print(json.dumps({"folder": str(target), "cases": [s["case"] for s in samples]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
