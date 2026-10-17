"""Small-GPU YOLO26n baseline, with local logs and post-training test evaluation."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT/"outputs/ultralytics_settings"))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--probe",action="store_true")
    p.add_argument("--imgsz",type=int,default=1024)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--nbs",type=int,default=64)
    p.add_argument("--patience",type=int,default=10)
    p.add_argument("--model",default="yolo26n.pt")
    p.add_argument("--skip-test",action="store_true")
    p.add_argument("--save-period",type=int,default=-1)
    p.add_argument("--lrf",type=float,default=.1)
    p.add_argument("--lr0",type=float,default=.001)
    p.add_argument("--warmup-bias-lr",type=float,default=.1)
    p.add_argument("--data",type=Path,default=ROOT/"data/yolo/pcbnet34_v1/dataset.yaml")
    p.add_argument("--device",default="0")
    p.add_argument("--name",default="pcbnet34_yolo26n_v1")
    a=p.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model_path=Path(a.model)
    if not model_path.is_absolute():
        model_path=ROOT/"weights"/model_path
    model=YOLO(str(model_path))
    peaks=[]
    def epoch_end(trainer):
        stats={"epoch":trainer.epoch+1,"allocated_peak_mib":round(torch.cuda.max_memory_allocated()/1024**2,1),
               "reserved_peak_mib":round(torch.cuda.max_memory_reserved()/1024**2,1)} if torch.cuda.is_available() else {"epoch":trainer.epoch+1}
        peaks.append(stats)
        (Path(trainer.save_dir)/"gpu_memory.json").write_text(json.dumps(peaks,indent=2),encoding="utf-8")
        (Path(trainer.save_dir)/"progress.json").write_text(json.dumps({"status":"training",**stats,"epochs_requested":a.epochs,
            "metrics":{k:float(v) for k,v in trainer.metrics.items()}},indent=2),encoding="utf-8")
        print("MEMORY_REPORT "+json.dumps(stats),flush=True)
    model.add_callback("on_train_epoch_end",epoch_end)
    results=model.train(data=str(a.data.resolve()),device=a.device,
        imgsz=a.imgsz,batch=a.batch,epochs=1 if a.probe else a.epochs,fraction=.1 if a.probe else 1.0,
        workers=0,amp=True,cache=False,project=str(ROOT/"runs"),name=a.name,exist_ok=False,
        optimizer="AdamW",lr0=a.lr0,lrf=a.lrf,weight_decay=.0005,warmup_epochs=2,warmup_bias_lr=a.warmup_bias_lr,
        seed=42,deterministic=True,patience=a.patience,plots=True,save=True,nbs=a.nbs,save_period=a.save_period,
        hsv_h=0,hsv_s=0,hsv_v=.1,degrees=0,translate=.05,scale=.2,shear=0,perspective=0,
        flipud=0,fliplr=0,mosaic=0,mixup=0,copy_paste=0)
    run=Path(model.trainer.save_dir)
    summary={"status":"probe_complete" if a.probe else "training_complete","best":str(run/"weights/best.pt"),
             "epochs_requested":1 if a.probe else a.epochs,"imgsz":a.imgsz,"batch":a.batch,
             "nbs":a.nbs,"lr0":a.lr0,"warmup_bias_lr":a.warmup_bias_lr,"initial_model":str(model_path),"data":str(a.data.resolve()),"epochs_completed":len(peaks),
             "validation":results.results_dict if results else None,"gpu_memory":peaks}
    if torch.cuda.is_available():
        summary["gpu_memory_final"]={"allocated_peak_mib":round(torch.cuda.max_memory_allocated()/1024**2,1),
            "reserved_peak_mib":round(torch.cuda.max_memory_reserved()/1024**2,1)}
    if not a.probe and not a.skip_test:
        best=YOLO(str(run/"weights/best.pt"))
        metrics=best.val(data=str(a.data.resolve()),split="test",device=a.device,
                         imgsz=a.imgsz,batch=a.batch,workers=0,plots=True,project=str(ROOT/"runs"),name=a.name+"_test")
        summary["test"]=metrics.results_dict
    (run/"baseline_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (run/"progress.json").write_text(json.dumps({"status":summary["status"],"epochs_completed":len(peaks),
        "epochs_requested":summary["epochs_requested"],"validation":summary["validation"]},indent=2),encoding="utf-8")
    print("BASELINE_RESULT "+json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
