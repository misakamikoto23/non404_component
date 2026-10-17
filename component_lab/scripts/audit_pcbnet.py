"""Inspect Dataset300 without changing the source annotations."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    categories, states, styles, fields = Counter(), Counter(), Counter(), Counter()
    issues, rows, text_examples = [], [], []
    hashes = defaultdict(list)
    overlaps = []
    prepared = Path(__file__).resolve().parents[1] / "data/prepared/manifest.json"
    old = json.loads(prepared.read_text(encoding="utf-8")) if prepared.exists() else {}
    old_hashes = {s["sha256"]: s for s in old.get("samples", [])}
    for path in sorted(args.source.glob("*/*_CVJsonStd.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            images = [f for f in path.parent.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if len(images) != 1:
                raise ValueError(f"image count {len(images)}")
            with Image.open(images[0]) as source:
                image = source.convert("RGB")
            w, h = image.size
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            hashes[digest].append(path.parent.name)
            if digest in old_hashes:
                overlaps.append({"pcbnet_case": path.parent.name, "official_case": old_hashes[digest]["case"], "official_split": old_hashes[digest]["split"]})
            canvas = data.get("canvas", {})
            if (canvas.get("width"), canvas.get("height")) != (w, h):
                issues.append({"case": path.parent.name, "reason": "canvas_mismatch", "canvas": canvas, "image_size": [w,h]})
            styles[str(data.get("nowCls"))] += 1
            components = data.get("components", {})
            for key, component in components.items():
                fields.update(component.keys())
                categories[str(component.get("category"))] += 1
                states[str(component.get("state"))] += 1
                box = component.get("realBBox") or component.get("location")
                if not box or not all(k in box for k in ("x1","y1","x2","y2")):
                    issues.append({"case": path.parent.name, "component": key, "reason": "missing_box"})
                    continue
                x1,y1,x2,y2 = (float(box[k]) for k in ("x1","y1","x2","y2"))
                if x2 <= x1 or y2 <= y1:
                    issues.append({"case": path.parent.name, "component": key, "reason": "nonpositive_box", "box": box})
                elif not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    issues.append({"case": path.parent.name, "component": key, "reason": "out_of_bounds", "box": box, "image_size": [w,h]})
            if len(text_examples)<3:
                text_examples.append({"case": path.parent.name, "top_level_keys": list(data), "component": next(iter(components.values()), None),
                                      "text_example": {k: str(v)[:1600] for k,v in data.items() if "text" in k.lower()}})
            rows.append({"case":path.parent.name,"size":[w,h],"components":len(components),"image":str(images[0]),"sha256":digest})
            if path.parent.name in {"0001","0002","0201","0300"}:
                previews=[]
                for flip in (False,True):
                    preview=image.copy()
                    draw=ImageDraw.Draw(preview)
                    for component in components.values():
                        box=component.get("realBBox") or component.get("location")
                        if not box: continue
                        x1,y1,x2,y2=(float(box[k]) for k in ("x1","y1","x2","y2"))
                        if flip: y1,y2=h-y2,h-y1
                        if x1<x2 and y1<y2: draw.rectangle([x1,y1,x2,y2],outline="red",width=2)
                    previews.append(ImageOps.contain(preview,(750,750)))
                combined=Image.new("RGB",(1500,780),"white")
                draw=ImageDraw.Draw(combined)
                draw.text((5,5),"LEFT: top-left origin    RIGHT: bottom-left origin",fill="black")
                combined.paste(previews[0],(0,25)); combined.paste(previews[1],(750,25))
                combined.save(args.output/(path.parent.name+"_coordinate_check.png"))
        except Exception as exc:
            issues.append({"case":path.parent.name,"reason":"read_error","error":str(exc)})
    report={"cases":len(rows),"components":sum(categories.values()),"categories":dict(categories),"states":dict(states),"styles":dict(styles),
            "component_fields":dict(fields),"issues":issues,"issue_counts":dict(Counter(i["reason"] for i in issues)),
            "duplicate_groups":[v for v in hashes.values() if len(v)>1],"overlap_with_official":overlaps,
            "examples":text_examples,"samples":rows}
    (args.output/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("cases","components","categories","states","issue_counts","duplicate_groups")},ensure_ascii=False))
    print("Official image overlap:",len(overlaps))


if __name__=="__main__": main()
