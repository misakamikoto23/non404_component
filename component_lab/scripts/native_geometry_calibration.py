"""Independent diagnostic gates for native-source/page registration, never GT export.

Reads existing extraction products and original source/image; writes a NEW subdir.
Source-wire color controls and held-out wire segments avoid the first pilot's
green-only failure mode. No renderer, training, paid API, or source modifications.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree

import native_schematic_label_pilot as native


def wire_color(node):
    color=native.child(native.child(node,"stroke"),"color")
    if len(color)>=5 and float(color[4])>0:
        return tuple(round(float(v)) for v in color[1:4])
    return "default_green"


def color_mask(arr, color):
    """Match anti-aliased stroke mixtures with the estimated page background."""
    arr=arr.astype(np.float32)
    if color=="default_green":
        return (arr[:,:,1]>arr[:,:,0]+22)&(arr[:,:,1]>arr[:,:,2]+22)&(arr[:,:,1]>55)
    background=np.median(arr.reshape(-1,3)[::max(1,arr.size//150000)],axis=0)
    target=np.array(color,dtype=np.float32)
    direction=target-background
    norm=float(direction@direction)
    if norm<100:
        return np.zeros(arr.shape[:2],dtype=bool)
    alpha=((arr-background)*direction).sum(axis=2)/norm
    residual=np.linalg.norm(arr-(background+alpha[:,:,None]*direction),axis=2)/math.sqrt(3)
    return (alpha>=.15)&(alpha<=1.15)&(residual<=27)


def distance_field(mask):
    return cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)


def sample_distance(field, points):
    points=np.asarray(points,dtype=float)
    if not len(points):
        return np.empty(0)
    return map_coordinates(field,[points[:,1],points[:,0]],order=1,mode="constant",cval=50)


def metric(distances):
    d=np.asarray(distances)
    if not len(d):
        return {"samples":0,"within_1_5px":None,"median_distance_px":None,"p95_distance_px":None}
    return {"samples":len(d),"within_1_5px":float((d<=1.5).mean()),"median_distance_px":float(np.median(d)),"p95_distance_px":float(np.percentile(d,95))}


def wire_points(root, sx, sy):
    fit=defaultdict(list);heldout=defaultdict(list);counts=Counter();endpoints=[]
    for node in native.children(root,"wire"):
        vertices=[(float(p[1]),float(p[2])) for p in native.children(native.child(node,"pts"),"xy")]
        if len(vertices)<2:
            continue
        endpoints.extend(vertices)
        color=wire_color(node)
        # Whole segments stay in a single partition; no fit/eval point interleaving.
        identity=json.dumps(vertices,separators=(",",":"))
        partition="fit" if int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)%2 else "heldout"
        target=fit if partition=="fit" else heldout
        counts[partition+"_segments"]+=1
        for a,b in zip(vertices,vertices[1:]):
            length=math.hypot((b[0]-a[0])*sx,(b[1]-a[1])*sy)
            if length<4:
                continue
            n=max(3,min(35,int(length/4)))
            for t in np.linspace(.15,.85,n):
                target[color].append(((a[0]+t*(b[0]-a[0]))*sx,(a[1]+t*(b[1]-a[1]))*sy))
    return fit,heldout,counts,endpoints


def warped(points,parameters,center):
    p=np.asarray(points,dtype=float)
    if not len(p):
        return p.reshape(-1,2)
    dx,dy,kx,ky=parameters
    return (p-np.asarray(center))*np.asarray([kx,ky])+np.asarray(center)+np.asarray([dx,dy])


def registration(root,arr,paper,allow_fit=False):
    h,w=arr.shape[:2];sx,sy=w/paper[0],h/paper[1];center=[w/2,h/2]
    fit,heldout,counts,endpoints=wire_points(root,sx,sy)
    fields={c:distance_field(color_mask(arr,c)) for c in set(fit)|set(heldout)}
    def distances(group,parameters=(0,0,1,1),control_offset=None):
        result=[]
        for color,pts in group.items():
            pos=warped(pts,parameters,center)
            if control_offset is not None:
                pos=pos+np.array(control_offset)
            result.extend(sample_distance(fields[color],pos))
        return np.asarray(result)
    baseline_fit=metric(distances(fit));baseline_eval=metric(distances(heldout))
    params=np.array([0.,0.,1.,1.]);optimization="skipped_paper_transform_sufficient"
    if allow_fit and baseline_fit["samples"]>=50 and (baseline_fit["within_1_5px"] or 0)<.97:
        def objective(x):
            d=distances(fit,x)
            return float(np.minimum(d,12).mean())+.01*(abs(x[0])+abs(x[1]))
        result=differential_evolution(objective,[(-12,12),(-12,12),(.985,1.015),(.985,1.015)],seed=0,maxiter=24,popsize=6,polish=False,tol=.01)
        if objective(result.x)+.1<objective(params):
            params=result.x;optimization="bounded_fit_on_fit_segments_only"
        else:
            optimization="bounded_fit_no_material_improvement"
    final_eval=metric(distances(heldout,params))
    controls=[metric(distances(heldout,params,c))["within_1_5px"] for c in ((13,0),(-13,0),(0,13),(0,-13))]
    valid_controls=[x for x in controls if x is not None]
    negative_mean=sum(valid_controls)/len(valid_controls) if valid_controls else None
    supported=bool(final_eval["samples"]>=50 and counts["heldout_segments"]>=5 and final_eval["within_1_5px"]>=.95 and negative_mean is not None and final_eval["within_1_5px"]-negative_mean>=.25 and abs(sx/sy-1)<.01)
    return {"paper_scales":[sx,sy],"source_wire_colors":[str(c) for c in fields],"segment_partition_counts":dict(counts),
            "baseline_fit":baseline_fit,"baseline_heldout":baseline_eval,"final_heldout":final_eval,
            "correction_dx_dy_kx_ky":params.tolist(),"optimization":optimization,"negative_shift_13px_within_1_5px_mean":negative_mean,
            "registration_supported":supported,"production_label_eligible":False},endpoints,params,fields


def raw_visible_pins(root):
    libraries={s[1]:s for s in native.children(native.child(root,"lib_symbols"),"symbol")}
    for symbol in native.children(root,"symbol"):
        lib=libraries.get(native.scalar(symbol,"lib_id"))
        if not lib or native.child(lib,"extends"):
            continue
        loc=native.child(symbol,"at");at=[float(loc[1]),float(loc[2]),float(loc[3]) if len(loc)>3 else 0]
        unit=int(native.scalar(symbol,"unit",1));style=int(native.scalar(symbol,"convert",native.scalar(symbol,"body_style",1)))
        mirror=native.scalar(symbol,"mirror","")
        for part in native.children(lib,"symbol"):
            match=re.search(r"_(\d+)_(\d+)$",part[1])
            if not match or int(match[1]) not in (0,unit) or int(match[2]) not in (0,style):
                continue
            for pin in native.children(part,"pin"):
                if "hide" in pin or native.child(pin,"hide"):
                    continue
                loc=native.child(pin,"at");x,y,angle=map(float,loc[1:4]);length=float(native.scalar(pin,"length",0))
                if length<=0:
                    continue
                end=(x+length*math.cos(math.radians(angle)),y+length*math.sin(math.radians(angle)))
                yield {"at":at,"mirror":mirror,"start":(x,y),"end":end,"number":native.scalar(pin,"number"),"length":length,
                       "dnp":native.scalar(symbol,"dnp")=="yes"}


def variant_transform(point,pin,variant):
    at=list(pin["at"]);mirror=pin["mirror"]
    if variant in ("rotation_sign_reversed","both_reversed"):
        at[2]=-at[2]
    if variant in ("mirror_axes_swapped","both_reversed"):
        mirror={"x":"y","y":"x"}.get(mirror,mirror)
    return native.transform_point(point,at,mirror)


def pin_diagnostics(root,arr,registration_data,endpoints,params):
    h,w=arr.shape[:2];sx,sy=registration_data["paper_scales"]
    a=arr.astype(np.int16)
    red=(a[:,:,0]>a[:,:,1]+22)&(a[:,:,0]>a[:,:,2]+22)&(a[:,:,0]>55)
    # KiCad dims DNP symbols to neutral gray; do not interpret that as a bad transform.
    gray=(a.max(axis=2)-a.min(axis=2)<25)&(a.max(axis=2)<225)&(a.min(axis=2)>45)
    fields={False:distance_field(red),True:distance_field(gray)}
    contacts=list(endpoints)
    for kind in ("junction","no_connect","label","global_label","hierarchical_label"):
        contacts.extend(native.point(n) for n in native.children(root,kind))
    tree=cKDTree(np.array(contacts)) if contacts else None
    groups=defaultdict(list);variants=defaultdict(list);counts=Counter()
    for pin in raw_visible_pins(root):
        counts["visible_nonzero_pins"]+=1
        counts["dnp_visible_nonzero_pins"]+=pin["dnp"]
        group="mirrored" if pin["mirror"] else "unmirrored"
        angle_group="quarter_turn" if abs(pin["at"][2]%180)>1e-5 else "zero_or_half_turn"
        points=[]
        for t in (.2,.4,.6,.8):
            local=(pin["start"][0]+t*(pin["end"][0]-pin["start"][0]),pin["start"][1]+t*(pin["end"][1]-pin["start"][1]))
            points.append(local)
        def distances(variant):
            mm=[variant_transform(p,pin,variant) for p in points]
            pixels=[[x*sx,y*sy] for x,y in mm]
            return sample_distance(fields[pin["dnp"]],warped(pixels,params,[w/2,h/2]))
        current=distances("current")
        for key in ("all",group,angle_group,"dnp_gray" if pin["dnp"] else "active_red"):
            groups[key].extend(current)
        current_contact=variant_transform(pin["start"],pin,"current")
        if tree is not None:
            counts["pin_connections_on_source_wire_label_nc"]+=tree.query(current_contact)[0]<=.0254
        for variant in ("rotation_sign_reversed","mirror_axes_swapped","both_reversed"):
            alternate=variant_transform(pin["start"],pin,variant)
            if math.dist(alternate,current_contact)>.2:
                variants[variant+"_current"].extend(current)
                variants[variant+"_alternate"].extend(distances(variant))
                if tree is not None:
                    counts[variant+"_eligible_connections"]+=1
                    counts[variant+"_current_source_contacts"]+=tree.query(current_contact)[0]<=.0254
                    counts[variant+"_alternate_source_contacts"]+=tree.query(alternate)[0]<=.0254
    stats={k:metric(v) for k,v in groups.items()}
    comparisons={k:metric(v) for k,v in variants.items()}
    return {"counts":dict(counts),"image_pin_line_alignment":stats,"deliberately_wrong_transform_controls":comparisons,
            "pin_alignment_supported":bool(stats.get("all",{}).get("samples",0)>=40 and (stats["all"]["within_1_5px"] or 0)>=.9),
            "caveat":"Theme-specific red active-pin/neutral-gray DNP-pin masks. Unconnected pins legitimately lack wire/label contacts. Wrong transforms may overlap symmetric pin grids. No human/renderer truth claimed."}


def run(args):
    extraction=args.extraction.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    paths=sorted((extraction/"sources").glob("*.json"))[:args.limit]
    rows=[];totals=Counter();pin_distance=defaultdict(list)
    for i,path in enumerate(paths):
        data=json.loads(path.read_text(encoding="utf-8"));source=Path(data["source_file"]);image=Path(data["image_file"])
        try:
            if native.sha256(source)!=data["source_sha256"] or native.sha256(image)!=data["image_sha256"]:
                raise ValueError("source_or_image_hash_mismatch")
            root=native.parse_sexpr(source.read_text(encoding="utf-8-sig"));arr=np.asarray(Image.open(image).convert("RGB"))
            paper=native.paper_size(root)
            if not paper:
                raise ValueError("unknown_paper")
            reg,endpoints,params,fields=registration(root,arr,paper,args.allow_fit)
            pins=pin_diagnostics(root,arr,reg,endpoints,params)
            record={"id":data["id"],"source_sha256":data["source_sha256"],"image_sha256":data["image_sha256"],"registration":reg,"pins":pins,
                    "production_label_eligible":False,"status":"independent_geometry_diagnostics_only"}
            native.write_json(output/(data["id"]+".json"),record)
            totals["images"]+=1;totals["registration_supported"]+=reg["registration_supported"];totals["pin_alignment_supported"]+=pins["pin_alignment_supported"]
            totals["no_wire_samples"]+=reg["final_heldout"]["samples"]==0
            totals["both_supported"]+=reg["registration_supported"] and pins["pin_alignment_supported"]
            for key,stats in pins["image_pin_line_alignment"].items():
                pin_distance[key].append((stats["samples"],stats["within_1_5px"] or 0))
            for k,v in pins["counts"].items():
                totals[k]+=v
            rows.append({"id":data["id"],"registration_supported":reg["registration_supported"],"wire_heldout_fraction":reg["final_heldout"]["within_1_5px"],"pin_fraction":pins["image_pin_line_alignment"].get("all",{}).get("within_1_5px"),"wire_points":reg["final_heldout"]["samples"],"pin_alignment_supported":pins["pin_alignment_supported"],"optimization":reg["optimization"]})
        except Exception as exc:
            totals["failed"]+=1;rows.append({"id":data["id"],"error":type(exc).__name__+":"+str(exc)[:200]})
        if (i+1)%10==0:
            print(json.dumps({"processed":i+1,"counts":dict(totals)}),flush=True)
    weighted={key:{"sample_points":sum(n for n,p in values),"within_1_5px_fraction":sum(n*p for n,p in values)/sum(n for n,p in values)} for key,values in pin_distance.items() if sum(n for n,p in values)}
    summary={"counts":dict(totals),"images":rows,"weighted_pin_pixel_diagnostics":weighted,"production_yolo_labels":0,"training_modified":False,"paid_calls":0,
             "gates":{"wire_heldout_min_points":50,"wire_heldout_min_segments":5,"wire_within_1_5px_fraction":.95,"shift_control_gap_min":.25,"pin_min_points":40,"pin_within_1_5px_fraction":.9},
             "remaining_blockers":["No independent KiCad SVG/renderer geometry for body bounding-box verification","Source-library identity still needs competition ontology mapping","Missing libraries, legacy arcs, hidden text/decorations and hierarchy unresolved","Fork/near-duplicate and prior-data overlap audit not completed"],
             "meaning":"Registration and pin-line diagnostics are automatic consistency checks; they do not prove label semantics, exhaustive detections, symbol bbox correctness or actual accuracy."}
    native.write_json(output/"SUMMARY.json",summary)
    table="\n".join(f"|{r['id']}|{r.get('wire_points','-')}|{r.get('wire_heldout_fraction','-')}|{r.get('pin_fraction','-')}|{r.get('registration_supported',False)}|{r.get('pin_alignment_supported',False)}|" for r in rows)
    report=f'''# 原生几何增强校准诊断\n\n在先前提取结果之外独立输出；未改源文件、审核或训练数据。\n\n- {totals['images']}页完成，失败{totals['failed']}页。\n- {totals['registration_supported']}页通过独立线段holdout像素对齐，{totals['no_wire_samples']}页无可用holdout导线采样。显式wire颜色（黑/粉/红/蓝等）单独建掩膜，避免把所有线假设成绿色。\n- {totals['pin_alignment_supported']}页通过可见pin线与渲染红色线像素的对应诊断；{totals['both_supported']}页两项都通过。\n- 另外比较了故意反转旋转符号、交换镜像轴的对照，并检查pin端点在源导线/标签/悬空标记上的命中数。未连接pin允许没有接触证据，不能据此删pin。\n- 默认只检验纸张变换，fit/eval按整段wire确定性分开，不把同一线段采样点拆到两侧。可选`--allow-fit`只用fit线段优化有限缩放/平移，heldout用于独立检查。\n- **生产标注0，训练修改0，API调用0。诊断通过仍不等于bbox真值或准确率。**\n\n## 建议的全自动质量门\n\n1. 源/图哈希和元件库依赖完整；未知/缺失依赖不生成完整监督。\n2. 项目及fork/近重复分组，隔离原验证/测试与已审核图片。\n3. 纸张/视口投影通过holdout导线与旋转/镜像pin对齐，同时负移位对照明显变差。\n4. 独立SVG渲染验证完整符号可见轮廓，统一bbox含引脚/不含文字的口径。当前缺此步，不能升级为生产标签。\n5. 库ID与结构映射题目类型，未映射类别隔离；保留位号、值、unit、引脚名与来源，Value不能无条件称型号。\n\n|样本|holdout点|导线命中率|pin线命中率|导线支持|pin支持|\n|---|---:|---:|---:|---|---|\n{table}\n\n机器细节见[SUMMARY.json](SUMMARY.json)及逐图JSON。\n'''
    (output/"REPORT.md").write_text(report,encoding="utf-8")
    with (output/"REPORT.md").open("a",encoding="utf-8") as handle:
        handle.write("\n## 独立性与支持范围\n\n这里的heldout是同一对源文件/已有PNG中的**独立导线线段**，按整段分开；不是独立渲染、不是第二套人工真值。可选拟合只读取fit线段，本次运行未启用拟合。通过诊断仅说明已支持的可见几何与现有渲染在像素位置上相容。\n\n隐藏引脚/文字未参与像素门槛；缺少嵌入库、无法解析的旧式圆弧、引脚装饰、非正交圆弧/圆、库继承、未解析的多层子页与未验证body-style仍须隔离。即使一个页面有足够正确导线和pin点，也不能把这些未支持对象自动升级为可靠框。\n\n")
        handle.write("\n## 本次发现\n\n源文件中显式wire颜色导致绿色掩膜的假失败，已按源颜色检验。DNP元件的符号/引脚被渲染为灰色，不能按红色pin掩膜视作错位；本次按源dnp状态分别使用红/灰诊断。三个没有wire的页面、三个holdout采样不足的页面保留为证据不足，不能强行升级。\n\n以下是像素位置诊断，**不是元件识别、引脚身份或网表准确率**：\n\n")
        for key,stats in weighted.items():
            handle.write(f"- {key}：{stats['sample_points']}个pin线采样点，1.5px内匹配比例{stats['within_1_5px_fraction']:.4%}。\n")
    print(json.dumps({"counts":dict(totals),"summary":str(output/"SUMMARY.json")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    base=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction",type=Path,default=base/"outputs/unattended_20261016/native_labels")
    parser.add_argument("--output",type=Path,default=base/"outputs/unattended_20261016/native_labels/calibration_v2")
    parser.add_argument("--limit",type=int,default=50)
    parser.add_argument("--allow-fit",action="store_true")
    run(parser.parse_args())
