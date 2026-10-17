"""One bounded prediction-only local OCR reread of unresolved component crops.

Run using .venv-ocr. This script never opens labels or chooses ROI from GT.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"ground", "gnd", "power_port", "net_port", "port", "no_connect"}


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b):
    intersection = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def context_window(box, width, height, minimum_margin=24, fraction=.5, max_side=384):
    x1,y1,x2,y2 = box
    cx,cy = (x1+x2)/2,(y1+y2)/2
    span_x = min(max_side, math.ceil(x2-x1+2*max(minimum_margin,(x2-x1)*fraction)))
    span_y = min(max_side, math.ceil(y2-y1+2*max(minimum_margin,(y2-y1)*fraction)))
    left,top = math.floor(cx-span_x/2),math.floor(cy-span_y/2)
    return [max(0,left),max(0,top),min(width,left+span_x),min(height,top+span_y)]


def transform_polygon(polygon, window, scaled_size):
    sx = scaled_size[0] / (window[2]-window[0])
    sy = scaled_size[1] / (window[3]-window[1])
    return [[float(x)/sx+window[0],float(y)/sy+window[1]] for x,y in polygon]


def merge_exact_overlaps(original, new, threshold=.5):
    """Keep originals and preserve distinct strings; never normalize or spell-correct."""
    kept = deepcopy(original)
    added = []
    for row in sorted(new, key=lambda row: -row["score"]):
        if any(row["raw"] == old.get("raw",old.get("text")) and overlap(row["bbox"],old["bbox"]) > threshold for old in kept):
            continue
        kept.append(row)
        added.append(row)
    used_ids = {row.get("id") for row in original}
    index = 1
    for row in added:
        while f"local_text_{index:04d}" in used_ids:
            index += 1
        row["id"] = f"local_text_{index:04d}"
        used_ids.add(row["id"])
        index += 1
    return kept,added


def self_test():
    import unittest
    class GeometryTests(unittest.TestCase):
        def test_original_coordinates(self):
            self.assertEqual(transform_polygon([[0,0],[30,60]],[100,200,120,230],(60,90)),[[100.,200.],[110.,220.]])
        def test_crop_clipped(self):
            self.assertEqual(context_window([0,0,20,20],100,100),[0,0,44,44])
            box=context_window([0,0,1000,1000],1000,1000)
            self.assertEqual(box,[308,308,692,692])
            box=context_window([.2,.2,1000.9,1000.9],1100,1100)
            self.assertLessEqual(box[2]-box[0],384)
        def test_raw_overlap_only(self):
            original=[{"id":"text_0001","raw":"10K","bbox":[0,0,10,10],"score":.8}]
            new=[{"raw":"10K","bbox":[0,0,10,10],"score":.9},
                 {"raw":"10k","bbox":[0,0,10,10],"score":.9},
                 {"raw":"10K","bbox":[20,20,30,30],"score":.9}]
            kept,added=merge_exact_overlaps(original,new)
            self.assertEqual(len(kept),3)
            self.assertEqual(len(added),2)
            self.assertEqual(kept[0],original[0])
            self.assertNotIn("id",new[0])
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,default=ROOT/"outputs/component_identity_round_001_20261017/inputs")
    parser.add_argument("--associations",type=Path,default=ROOT/"outputs/component_identity_round_001_20261017/experiment_v1/identity_v1/associations")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/component_identity_round_001_20261017/local_ocr_v1")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output.exists():
        raise FileExistsError("output already exists; no automatic retry or repeated experiment")
    manifest=read(args.inputs/"manifest.json")
    if not manifest.get("complete") or len(manifest["jobs"])!=20:
        raise ValueError("requires frozen complete DEV20 manifest")
    from PIL import Image
    import numpy as np
    from rapidocr import RapidOCR,OCRVersion,ModelType
    settings={"method":"single_targeted_local_reference_reread", "scale":3.,"max_rois_per_case":4,"max_rois_total":80,
              "minimum_margin":24,"axis_margin_fraction":.5,"max_context_side":384,"duplicate_iou_threshold":.5,
              "selection":"unresolved reference; exclude ground/power/net ports; confidence descending then source ID",
              "selection_uses_labels":False,"paid_api_calls":0,"retries":0,"text_score":.35,
              "package_version":importlib.metadata.version("rapidocr"),
              "model_sha256":{p.name:sha(p) for p in sorted((ROOT/"weights/ocr_rapid").glob("*.onnx"))}}
    started=time.perf_counter()
    engine=RapidOCR(params={"Global.model_root_dir":str(ROOT/"weights/ocr_rapid"),
                           "Global.max_side_len":4096,"Global.text_score":.35,
                           "Det.ocr_version":OCRVersion.PPOCRV5,"Det.model_type":ModelType.MOBILE,
                           "Rec.ocr_version":OCRVersion.PPOCRV5,"Rec.model_type":ModelType.MOBILE,
                           "Det.limit_side_len":2048,"Det.limit_type":"max",
                           "EngineConfig.onnxruntime.intra_op_num_threads":2,
                           "EngineConfig.onnxruntime.inter_op_num_threads":1,
                           "EngineConfig.onnxruntime.use_cuda":False})
    args.output.mkdir(parents=True)
    completed=[]
    output_jobs=[]
    total_rois=0
    save(args.output/"config.json",settings)
    for job in manifest["jobs"]:
        case=job["id"]
        case_started=time.perf_counter()
        association_path=args.associations/f"{case}.json"
        association=read(association_path)
        original_cache=read(job["ocr"])
        candidates=[row for row in association["components"]
                    if not row.get("reference") and row.get("type") not in EXCLUDE]
        candidates=sorted(candidates,key=lambda row:(-row.get("detection",{}).get("score",0.),row.get("id","")))[:4]
        new=[]
        roi_log=[]
        with Image.open(job["image"]) as opened:
            image=opened.convert("RGB")
        for rank,component in enumerate(candidates,1):
            if total_rois>=80:
                break
            window=context_window(component["bbox"],image.width,image.height)
            crop=image.crop(window)
            scaled=crop.resize((crop.width*3,crop.height*3),Image.Resampling.LANCZOS)
            roi_path=args.output/case/f"roi_{rank:02d}.png"
            roi_path.parent.mkdir(parents=True,exist_ok=True)
            crop.save(roi_path)
            roi_started=time.perf_counter()
            result=engine(np.asarray(scaled))
            roi_seconds=time.perf_counter()-roi_started
            roi_record={"rank":rank,"component_id":component["id"],"detector_score":component.get("detection",{}).get("score"),
                        "component_type":component.get("type"),"component_bbox":component["bbox"],"window":window,
                        "scaled_size":list(scaled.size),"roi_sha256":sha(roi_path),"seconds":roi_seconds,"raw_texts":0}
            if result.boxes is not None:
                for polygon,text,score in zip(result.boxes,result.txts,result.scores):
                    transformed=transform_polygon(polygon.tolist(),window,scaled.size)
                    xs,ys=zip(*transformed)
                    bbox=[min(xs),min(ys),max(xs),max(ys)]
                    if bbox[2]<=bbox[0] or bbox[3]<=bbox[1]:
                        continue
                    new.append({"text":str(text),"raw":str(text),"score":float(score),"bbox":bbox,"polygon":transformed,
                                "orientation":math.degrees(math.atan2(transformed[1][1]-transformed[0][1],transformed[1][0]-transformed[0][0])),
                                "source_view":f"local_roi_{rank:02d}","source_window":window,
                                "provenance":{"method":settings["method"],"component_id":component["id"],"scale":3.,
                                              "roi_sha256":roi_record["roi_sha256"],"original_image_sha256":sha(job["image"])}})
                    roi_record["raw_texts"]+=1
            roi_log.append(roi_record)
            total_rois+=1
        merged,added=merge_exact_overlaps(original_cache["texts"],new)
        record={"id":case,"rois":len(roi_log),"raw_new_texts":len(new),"added_texts":len(added),
                "base_texts":len(original_cache["texts"]),"merged_texts":len(merged),"seconds":time.perf_counter()-case_started,
                "image_sha256":sha(job["image"]),"detections_sha256":sha(job["detections"]),
                "association_sha256":sha(association_path),"base_ocr_sha256":sha(job["ocr"]),"roi_details":roi_log}
        extended=deepcopy(original_cache)
        extended["texts"]=merged
        extended["local_refinement"]={"settings":settings,"record":record,"ground_truth_used":False}
        target=args.output/case/"ocr_rapidocr.json"
        save(target,extended)
        output_jobs.append({**job,"ocr":str(target.resolve())})
        completed.append(record)
        save(args.output/"progress.json",{"complete":False,"jobs":completed,"rois":total_rois})
        print(f"{case}: {len(roi_log)} ROIs, {len(new)} raw texts, {len(added)} added, {record['seconds']:.2f}s",flush=True)
    total={"cases":len(completed),"rois":total_rois,"raw_new_texts":sum(r["raw_new_texts"] for r in completed),
           "added_texts":sum(r["added_texts"] for r in completed),"seconds":time.perf_counter()-started}
    save(args.output/"manifest.json",{"jobs":output_jobs,"complete":True,"source_manifest_sha256":sha(args.inputs/"manifest.json"),
                                        "kind":"prediction_only_local_ocr_DEV20","settings":settings,"totals":total})
    save(args.output/"progress.json",{"complete":True,"jobs":completed,"totals":total,"settings":settings})
    lines=["# Targeted local reference OCR — DEV20", "", "One bounded prediction-only intervention; no GT reads, API calls, retries, training, or original cache changes.", ""]
    lines += [f"- {k}: {v}" for k,v in total.items()]
    lines += ["", "Selection: up to four unresolved-reference physical/generic detections per case, ordered by detector confidence descending then source ID. Ground, power/network port and no-connect families excluded.",
              "", "Each crop is reread once at 3x, with max(24 px, 50% component axis) context margins and a 384 original-pixel context-side cap. Raw OCR strings remain unchanged. Original tokens are retained; new exact-raw tokens with IoU > 0.5 against a retained token are dropped. Distinct strings and nonoverlapping repeated strings remain.",
              "", "Counts represent OCR availability, not accuracy. Scoring is a separate evaluator operation. The manifest preserves original image/detection paths and points only OCR paths at extended caches.",
              "", "```powershell", ".\\.venv-ocr\\Scripts\\python.exe scripts/refine_local_reference_ocr.py --self-test", ".\\.venv-ocr\\Scripts\\python.exe scripts/refine_local_reference_ocr.py", "```", "", "Output directory must not already exist. Per-ROI timing/provenance are in progress.json; model and source hashes are recorded."]
    (args.output/"REPORT.md").write_text("\n".join(lines),"utf-8")
    print(json.dumps(total,ensure_ascii=True),flush=True)


if __name__=="__main__":
    main()
