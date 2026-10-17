"""Prepare a separate 34-class baseline while preserving existing holdouts."""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image


def prepare(source, output, previous):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; previous exports will not be overwritten.")
    old = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
    old_splits = {r["sha256"]: r["split"] for r in old.get("samples", [])}
    unique, duplicates, fixes = {}, [], []
    for path in sorted(source.glob("*/*_CVJsonStd.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        images = [p for p in path.parent.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if len(images) != 1:
            raise ValueError(f"Image pairing failed: {path}")
        with Image.open(images[0]) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            digest = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        if digest in unique:
            duplicates.append({"case": path.parent.name, "kept": unique[digest]["case"]})
            continue
        labels = []
        for component_id, component in annotation["components"].items():
            if component.get("state") == "deleted":
                fixes.append({"case": path.parent.name, "component": component_id, "action": "exclude_deleted"})
                continue
            box = component.get("realBBox") or component.get("location")
            raw = [float(box[k]) for k in ("x1", "y1", "x2", "y2")]
            x1,y1,x2,y2 = [max(0,min(limit,v)) for limit,v in zip([width,height,width,height],raw)]
            if [x1,y1,x2,y2] != raw:
                fixes.append({"case": path.parent.name, "component": component_id, "action": "clip", "original": raw})
            if x2<=x1 or y2<=y1:
                raise ValueError(f"Empty box: {path}/{component_id}")
            labels.append({"category": component["category"], "bbox": [x1,y1,x2,y2], "component_id":component_id})
        unique[digest] = {"case":path.parent.name,"image":images[0],"annotation":path,"sha256":digest,
                          "width":width,"height":height,"labels":labels,"split":old_splits.get(digest)}
    rows=list(unique.values())
    fresh=[r for r in rows if r["split"] is None]
    random.Random(42).shuffle(fresh)
    # Keep examples of each new category in train before assigning extra holdouts.
    known={l["category"] for r in rows if r["split"]=="train" for l in r["labels"]}
    candidates=[]
    for row in fresh:
        categories={l["category"] for l in row["labels"]}
        if categories-known:
            row["split"]="train"
            known.update(categories)
        else:
            candidates.append(row)
    nt=max(1,round(len(fresh)*.1))
    nv=max(1,round(len(fresh)*.1))
    for i,row in enumerate(candidates):
        row["split"]="test" if i<nt else "val" if i<nt+nv else "train"
    classes=sorted({l["category"] for r in rows for l in r["labels"]})
    counts={s:Counter() for s in ["train","val","test"]}
    for row in rows:
        split=row["split"]
        destination=output/"images"/split/(row["case"]+row["image"].suffix.lower())
        label_path=output/"labels"/split/(row["case"]+".txt")
        destination.parent.mkdir(parents=True,exist_ok=True)
        label_path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(row["image"],destination)
        lines=[]
        for label in row["labels"]:
            x1,y1,x2,y2=label["bbox"]
            w,h=row["width"],row["height"]
            lines.append(f"{classes.index(label['category'])} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
            counts[split][label["category"]]+=1
        label_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    yaml=f"path: {json.dumps(str(output.resolve()),ensure_ascii=False)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {json.dumps(classes,ensure_ascii=False)}\n"
    (output/"dataset.yaml").write_text(yaml,encoding="utf-8")
    summary={"source":str(source.resolve()),"origin":"top-left","classes":classes,"seed":42,"counts":counts,
             "splits":{s:[r["case"] for r in rows if r["split"]==s] for s in counts},"duplicates":duplicates,"fixes":fixes,
             "missing_train_classes":[c for c in classes if counts["train"][c]==0],
             "samples":[{k:str(v) if isinstance(v,Path) else v for k,v in r.items()} for r in rows]}
    assert not summary["missing_train_classes"],summary["missing_train_classes"]
    (output/"manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"diagrams":len(rows),"classes":len(classes),"splits":{s:len(v) for s,v in summary["splits"].items()},"fixes":len(fixes)},ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--previous",type=Path,required=True)
    a=p.parse_args()
    prepare(a.source,a.output,a.previous)
