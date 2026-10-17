"""Collapse original PCBnet classes to one detection class, retaining ResNet metadata."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,default=root/"data/yolo/pcbnet34_v1")
    p.add_argument("--output",type=Path,default=root/"data/yolo/pcbnet_component_v1")
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("Output directory must be empty")
    manifest=json.loads((a.source/"manifest.json").read_text(encoding="utf-8"))
    for row in manifest["samples"]:
        if not row["annotation"].endswith("_CVJsonStd.json"):
            raise ValueError("Only original standard PCBnet annotations are allowed")
        suffix=Path(row["image"]).suffix.lower()
        image=a.source/"images"/row["split"]/(row["case"]+suffix)
        old_label=a.source/"labels"/row["split"]/(row["case"]+".txt")
        destination=a.output/"images"/row["split"]/image.name
        label_path=a.output/"labels"/row["split"]/old_label.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        label_path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(image,destination)
        lines=["0 "+line.split(maxsplit=1)[1] for line in old_label.read_text().splitlines() if line.strip()]
        assert len(lines)==len(row["labels"])
        label_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest["original_category_names"]=manifest.pop("classes")
    manifest["original_category_counts"]=manifest.pop("counts")
    manifest["classes"]=["component"]
    manifest["counts"]={s:{"component":sum(len(r["labels"]) for r in manifest["samples"] if r["split"]==s)} for s in manifest["splits"]}
    manifest["task"]="single_class_component_localization"
    manifest["category_policy"]="All original active components map to detector class0; raw category retained per sample label for future ResNet. No user/AI labels used."
    manifest["missing_train_classes"]=[]
    (a.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    (a.output/"dataset.yaml").write_text(f"path: {json.dumps(str(a.output.resolve()),ensure_ascii=False)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: ['component']\n",encoding="utf-8")
    print(json.dumps({"classes":manifest["classes"],"images":{s:len(v) for s,v in manifest["splits"].items()},"boxes":manifest["counts"]},ensure_ascii=False))


if __name__=="__main__":main()
