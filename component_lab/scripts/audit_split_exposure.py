"""Read-only: list every exact decoded-pixel copy of the dev-split images under data/, outputs/ and runs/.

Usage: python -I scripts/audit_split_exposure.py <component_lab dir> > outputs/contest6_dev_split_exposure/exact_pixel_hits.json
"""
import hashlib, io, json, sys, zipfile
from collections import defaultdict
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
lab = Path(sys.argv[1])
split = json.loads((lab / "configs/contest6_dev_split.json").read_text(encoding="utf-8"))
wanted = {c["case"] for c in split["cases"]}
def sha(im):
    im = im.convert("RGB"); h = hashlib.sha256(f"{im.width}x{im.height}:RGB:".encode()); h.update(im.tobytes()); return h.hexdigest()
targets, sizes = {}, set()
with zipfile.ZipFile(lab.parent / "赛题六公开数据集.zip") as z:
    for n in z.namelist():
        if "/200_train_cases/" in n and n.lower().endswith((".png", ".jpg", ".jpeg")):
            case = n.split("/200_train_cases/")[1].split("/")[0]
            if case in wanted:
                with Image.open(io.BytesIO(z.read(n))) as im:
                    targets[sha(im)] = case; sizes.add(im.size)
hits = defaultdict(list)
for root in ("data", "outputs", "runs"):
    for p in (lab / root).rglob("*"):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg") or ".venv" in p.parts:
            continue
        try:
            with Image.open(p) as im:
                if im.size not in sizes: continue
                h = sha(im)
        except Exception:
            continue
        if h in targets:
            hits[targets[h]].append(str(p.relative_to(lab)).replace("\\", "/"))
json.dump({"cases": sorted(wanted), "hits": hits}, sys.stdout, ensure_ascii=False, indent=1)
