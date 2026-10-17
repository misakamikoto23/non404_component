"""Run the component pipeline on representative held-out test diagrams."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=root/"data/yolo/pcbnet_component_v1")
    parser.add_argument("--weights",type=Path,default=root/"weights/pcbnet_component_yolo26n_v1.pt")
    parser.add_argument("--output",type=Path,default=root/"outputs/pcbnet_component_predictions")
    parser.add_argument("--imgsz",type=int,default=1280)
    parser.add_argument("--device",default="0")
    parser.add_argument("--tile-size",type=int)
    parser.add_argument("--tile-overlap",type=float,default=.25)
    parser.add_argument("--tile-max-tiles",type=int,default=100)
    args=parser.parse_args()
    data_root=args.data_root.resolve()
    weights=args.weights.resolve()
    output_root=args.output.resolve()
    manifest=json.loads((data_root/"manifest.json").read_text(encoding="utf-8"))
    cases=sorted((r for r in manifest["samples"] if r["split"]=="test"),key=lambda r:len(r["labels"]))
    if not cases:
        raise ValueError("No held-out test cases in manifest")
    selected=[cases[0],cases[len(cases)//2],cases[-1]]
    results=[]
    for row in selected:
        source=data_root/"images/test"/(row["case"]+Path(row["image"]).suffix.lower())
        output=output_root/row["case"]
        command=[sys.executable,"-m","component_lab.cli","infer","--image",str(source),
                 "--yolo",str(weights),"--imgsz",str(args.imgsz),
                 "--device",args.device,"--output",str(output)]
        if args.tile_size is not None:
            command.extend(["--tile-size",str(args.tile_size),"--tile-overlap",str(args.tile_overlap),
                            "--tile-max-tiles",str(args.tile_max_tiles)])
        subprocess.run(command,check=True)
        report=json.loads((output/"components.json").read_text(encoding="utf-8"))
        results.append({"case":row["case"],"ground_truth_count":len(row["labels"]),"predicted_count":len(report["components"]),"overlay":str(output/"overlay.png")})
    output_root.mkdir(parents=True,exist_ok=True)
    (output_root/"summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(results,ensure_ascii=False))


if __name__=="__main__":main()
