"""Offline KiCad source extraction; diagnostic projections are NEVER training labels.

Only reads native data. Does not launch projects, fetch libraries, modify source,
or export YOLO annotations. Keeps the distinction between source facts, calculated
geometry, and image-registration diagnostics explicit in each record.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

PAPER_MM = {"A0": (1189, 841), "A1": (841, 594), "A2": (594, 420),
            "A3": (420, 297), "A4": (297, 210), "A5": (210, 148),
            "A": (279.4, 215.9), "B": (431.8, 279.4), "C": (558.8, 431.8),
            "D": (863.6, 558.8), "E": (1117.6, 863.6)}
TOKEN = re.compile(r'\s+|;[^\n]*|\(|\)|"(?:\\.|[^"\\])*"|[^\s();"]+')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_sexpr(text: str, max_depth: int = 128, max_tokens: int = 8_000_000) -> list:
    """Bounded data parser, no eval/pickle/XML entities/project execution."""
    stack: list[list] = []
    roots: list = []
    pos = 0
    for count, match in enumerate(TOKEN.finditer(text), 1):
        if count > max_tokens:
            raise ValueError("token_limit")
        if match.start() != pos:
            raise ValueError(f"invalid_token_at_{pos}")
        pos = match.end()
        token = match.group()
        if token.isspace() or token.startswith(";"):
            continue
        if token == "(":
            item: list = []
            (stack[-1] if stack else roots).append(item)
            stack.append(item)
            if len(stack) > max_depth:
                raise ValueError("depth_limit")
        elif token == ")":
            if not stack:
                raise ValueError("unmatched_close")
            stack.pop()
        else:
            if not stack:
                raise ValueError("atom_outside_expression")
            if token.startswith('"'):
                # KiCad escapes are decoded conservatively; preserve unknown escapes.
                token = re.sub(r'\\([\\"nrt])', lambda m: {"n": "\n", "r": "\r", "t": "\t"}.get(m[1], m[1]), token[1:-1])
            stack[-1].append(token)
    if pos != len(text) or stack or len(roots) != 1:
        raise ValueError("incomplete_or_multiple_expressions")
    return roots[0]


def children(node: list, key: str) -> list[list]:
    return [x for x in node[1:] if isinstance(x, list) and x and x[0] == key]


def child(node: list, key: str) -> list:
    return next(iter(children(node, key)), [])


def scalar(node: list, key: str, default: Any = None) -> Any:
    part = child(node, key)
    return part[1] if len(part) > 1 else default


def point(node: list, key: str = "at") -> tuple[float, float]:
    p = child(node, key)
    return (float(p[1]), float(p[2]))


def transform_point(p: tuple[float, float], at: list[float], mirror: str = "") -> tuple[float, float]:
    """Library Y-up -> page Y-down; stored angle is counterclockwise in Y-up.

    Mirror convention follows KiCad's axis tokens: x flips local y, y flips local x.
    Kept diagnostic until image overlays and registration are independently checked.
    """
    x, y = p
    if "x" in mirror:
        y = -y
    if "y" in mirror:
        x = -x
    angle = math.radians(at[2])
    return (at[0] + x * math.cos(angle) - y * math.sin(angle),
            at[1] - x * math.sin(angle) - y * math.cos(angle))


def paper_size(root: list) -> tuple[float, float] | None:
    p = child(root, "paper")
    if len(p) < 2:
        return None
    if p[1] in PAPER_MM:
        w, h = PAPER_MM[p[1]]
    elif p[1] == "User" and len(p) >= 4:
        w, h = float(p[2]), float(p[3])
    elif len(p) >= 3:
        try:
            w, h = float(p[1]), float(p[2])
        except ValueError:
            return None
    else:
        return None
    if "portrait" in p:
        w, h = min(w, h), max(w, h)
    return (w, h) if w > 0 and h > 0 else None


def arc_points(node: list) -> tuple[list[tuple[float, float]], bool]:
    """Exact centerline arc extrema from start/mid/end; collinear fallback flagged."""
    a, b, c = point(node, "start"), point(node, "mid"), point(node, "end")
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2 * (ax * (by-cy) + bx * (cy-ay) + cx * (ay-by))
    if abs(d) < 1e-10:
        return [a, b, c], False
    ux = ((ax*ax+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by))/d
    uy = ((ax*ax+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax-cx)+(cx*cx+cy*cy)*(bx-ax))/d
    radius = math.hypot(ax-ux, ay-uy)
    start, middle, end = [math.atan2(y-uy, x-ux) % (2*math.pi) for x, y in (a, b, c)]
    ccw = (middle-start) % (2*math.pi) <= (end-start) % (2*math.pi)
    pts = [a, b, c]
    for angle in (0, math.pi/2, math.pi, 3*math.pi/2):
        included = ((angle-start) % (2*math.pi) <= (end-start) % (2*math.pi)+1e-10) if ccw else ((start-angle) % (2*math.pi) <= (start-end) % (2*math.pi)+1e-10)
        if included:
            pts.append((ux+radius*math.cos(angle), uy+radius*math.sin(angle)))
    return pts, True


def symbol_geometry(lib: list, unit: int, style: int, at: list[float], mirror: str) -> dict:
    points: list[tuple[float, float]] = []
    pins = []
    issues = []
    primitive_counts: Counter = Counter()
    max_half_stroke = 0.0
    if child(lib, "extends"):
        return {"bbox_mm": None, "pins": [], "issues": ["inherited_library_not_resolved"], "primitive_counts": {}}
    for part in children(lib, "symbol"):
        match = re.search(r"_(\d+)_(\d+)$", part[1])
        if not match:
            issues.append("invalid_unit_suffix")
            continue
        p_unit, p_style = map(int, match.groups())
        if p_unit not in (0, unit) or p_style not in (0, style):
            continue
        for primitive in part[2:]:
            if not isinstance(primitive, list) or not primitive:
                continue
            kind = primitive[0]
            primitive_counts[kind] += 1
            pts = []
            try:
                if kind in ("polyline", "bezier"):
                    pts = [(float(p[1]), float(p[2])) for p in children(child(primitive, "pts"), "xy")]
                    if kind == "bezier":
                        issues.append("bezier_control_hull_bbox_conservative")
                elif kind == "rectangle":
                    x1, y1 = point(primitive, "start"); x2, y2 = point(primitive, "end")
                    pts = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
                elif kind == "circle":
                    x, y = point(primitive, "center"); r = float(scalar(primitive, "radius"))
                    pts = [(x-r, y), (x+r, y), (x, y-r), (x, y+r)]
                    # General non-90 rotations need transformed circle extrema instead.
                    if abs(at[2] % 90) > 1e-6:
                        issues.append("nonorthogonal_circle_projection")
                elif kind == "arc":
                    pts, exact = arc_points(primitive)
                    if not exact:
                        issues.append("degenerate_arc")
                    if abs(at[2] % 90) > 1e-6:
                        issues.append("nonorthogonal_arc_projection")
                elif kind == "pin":
                    p = child(primitive, "at"); x, y, angle = map(float, p[1:4])
                    length = float(scalar(primitive, "length", 0))
                    end = (x+length*math.cos(math.radians(angle)), y+length*math.sin(math.radians(angle)))
                    hidden = "hide" in primitive or bool(child(primitive, "hide"))
                    pins.append({"number": scalar(primitive, "number"), "name": scalar(primitive, "name"),
                                 "electrical_type": primitive[1], "shape": primitive[2], "hidden": hidden,
                                 "connection_mm": list(transform_point((x,y), at, mirror)),
                                 "inner_mm": list(transform_point(end, at, mirror)), "length_mm": length})
                    if not hidden:
                        pts = [(x, y), end]
                        if primitive[2] not in ("line", "non_logic"):
                            issues.append("pin_decorations_not_in_bbox")
                elif kind in ("text", "text_box"):
                    issues.append("embedded_symbol_text_not_in_bbox")
                else:
                    issues.append("unsupported_primitive_" + str(kind))
                if pts:
                    stroke = float(scalar(child(primitive, "stroke"), "width", 0))
                    if stroke == 0:
                        issues.append("default_stroke_width_not_resolved")
                    max_half_stroke = max(max_half_stroke, stroke/2)
                    points.extend(transform_point(p, at, mirror) for p in pts)
            except (ValueError, TypeError, IndexError):
                issues.append("malformed_primitive_" + str(kind))
    bbox = None
    if points:
        xs, ys = zip(*points)
        bbox = [min(xs)-max_half_stroke, min(ys)-max_half_stroke, max(xs)+max_half_stroke, max(ys)+max_half_stroke]
    return {"bbox_mm": bbox, "pins": pins, "issues": sorted(set(issues)), "primitive_counts": dict(primitive_counts)}


def extract_source(path: Path) -> dict:
    if path.suffix.lower() != ".kicad_sch":
        raise ValueError("unsupported_source_format")
    if path.stat().st_size > 64*1024*1024:
        raise ValueError("source_size_limit")
    root = parse_sexpr(path.read_text(encoding="utf-8-sig"))
    if not root or root[0] != "kicad_sch":
        raise ValueError("not_modern_kicad_schematic")
    libraries = {x[1]: x for x in children(child(root, "lib_symbols"), "symbol")}
    symbols = []
    for index, node in enumerate(children(root, "symbol")):
        lib_id = scalar(node, "lib_id", "")
        loc = child(node, "at")
        at = [float(loc[1]), float(loc[2]), float(loc[3]) if len(loc)>3 else 0.0]
        unit, style = int(scalar(node, "unit", 1)), int(scalar(node, "convert", scalar(node, "body_style", 1)))
        mirror = scalar(node, "mirror", "")
        props = {}
        for p in children(node, "property"):
            effects = child(p, "effects")
            hidden = "hide" in effects or scalar(effects, "hide", "no") == "yes"
            props[p[1]] = {"text": p[2], "position_mm": child(p, "at")[1:], "hidden": hidden}
        geometry = symbol_geometry(libraries[lib_id], unit, style, at, mirror) if lib_id in libraries else {"bbox_mm": None, "pins": [], "issues": ["missing_embedded_library"], "primitive_counts": {}}
        instance_paths = []
        for project in children(child(node, "instances"), "project"):
            for p in children(project, "path"):
                instance_paths.append({"project": project[1], "sheet_path": p[1], "reference": scalar(p,"reference"), "unit": scalar(p,"unit")})
        symbols.append({"index": index, "uuid": scalar(node,"uuid"), "lib_id": lib_id,
                        "reference": props.get("Reference",{}).get("text"),
                        "value": props.get("Value",{}).get("text"),
                        "part_number": None,
                        "part_number_note": "Value is source text, not a verified manufacturer part number",
                        "footprint": props.get("Footprint",{}).get("text"),
                        "unit": unit, "body_style": style, "at_mm": at, "mirror": mirror,
                        "source_kind": "power_symbol" if lib_id.startswith("power:") else "library_symbol",
                        "dnp": scalar(node,"dnp"), "in_bom": scalar(node,"in_bom"), "on_board": scalar(node,"on_board"),
                        "properties": props, "instance_paths": instance_paths, **geometry})
    wires = []
    for node in children(root,"wire"):
        pts = [[float(p[1]), float(p[2])] for p in children(child(node,"pts"),"xy")]
        if len(pts)>=2:
            wires.append(pts)
    sheets = [{"uuid":scalar(s,"uuid"), "properties":{p[1]:p[2] for p in children(s,"property")}} for s in children(root,"sheet")]
    return {"format":"kicad_sch", "source_version":scalar(root,"version"), "sheet_uuid":scalar(root,"uuid"),
            "paper_mm":paper_size(root), "symbols":symbols, "wires_mm":wires, "hierarchical_sheets":sheets,
            "junctions_mm":[list(point(n)) for n in children(root,"junction")],
            "no_connects_mm":[list(point(n)) for n in children(root,"no_connect")],
            "status":"source_facts_and_unvalidated_geometric_candidates_not_training_ground_truth",
            "bbox_convention":"symbol graphics and visible pin segments; excludes field text/pin text; pin decorations may be incomplete"}


def project_split(project: str) -> str:
    bucket = int(hashlib.sha256(project.lower().encode()).hexdigest()[:12],16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def project_groups(records: list[dict]) -> dict[str, str]:
    """Union same repository and exact source/image duplicates before splitting."""
    parent = {r["id"]:r["id"] for r in records}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    seen = {}
    for r in records:
        for key,value in (("repo",r.get("source_repository","" ).lower()), ("source",r.get("source_sha256")), ("rgb",r.get("rgb_sha256"))):
            if not value:
                continue
            ident=(key,value)
            if ident in seen:
                a,b=find(r["id"]),find(seen[ident]); parent[max(a,b)]=min(a,b)
            else:
                seen[ident]=r["id"]
    groups=defaultdict(list)
    for r in records:
        groups[find(r["id"])].append(r.get("source_repository") or r["id"])
    return {r["id"]:min(groups[find(r["id"])]).lower() for r in records}


def inside(path: Path, root: Path) -> Path:
    path = path.resolve(); root = root.resolve()
    if not path.is_relative_to(root):
        raise ValueError("source_path_outside_dataset")
    return path


def diagnostic_projection(data: dict, image_path: Path, overlay_path: Path | None = None) -> dict:
    """Paper-based transform plus independent wire alignment check, no fitting to GT."""
    import numpy as np
    from PIL import Image, ImageDraw
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image).astype(np.int16)
    w,h = image.size
    paper = data["paper_mm"]
    if not paper:
        return {"status":"no_known_paper_size", "production_label_eligible":False}
    sx,sy = w/paper[0],h/paper[1]
    ratio_error=abs(sx/sy-1)
    # KiCad default theme: useful registration diagnostic, NOT universal semantic mask.
    green = (arr[:,:,1] > arr[:,:,0]+25) & (arr[:,:,1] > arr[:,:,2]+25) & (arr[:,:,1] > 60)
    dark = arr.min(axis=2) < 180
    wire_pixels=[]
    for wire in data["wires_mm"]:
        for a,b in zip(wire,wire[1:]):
            length=math.dist(a,b)*min(sx,sy)
            n=max(2,min(80,int(length/3)))
            for i in range(1,n):
                t=i/n; wire_pixels.append(((a[0]+t*(b[0]-a[0]))*sx,(a[1]+t*(b[1]-a[1]))*sy))
    def hitrate(mask, offset=(0,0)):
        hits=0; total=0
        for x,y in wire_pixels:
            x,y=round(x+offset[0]),round(y+offset[1])
            if 1<=x<w-1 and 1<=y<h-1:
                total+=1; hits+=bool(mask[y-1:y+2,x-1:x+2].any())
        return hits/total if total else None
    chosen=green if green.sum()>100 else dark
    expected=hitrate(chosen)
    shifted=[hitrate(chosen,p) for p in ((13,0),(-13,0),(0,13),(0,-13))]
    shift_mean=sum(v for v in shifted if v is not None)/len([v for v in shifted if v is not None]) if any(v is not None for v in shifted) else None
    coords=[]
    for s in data["symbols"]:
        box=s["bbox_mm"]
        b=[box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy] if box else None
        coords.append({"index":s["index"],"reference":s["reference"],"bbox_pixels_candidate":b,
                       "pins_pixels":[{"number":p["number"],"connection":[p["connection_mm"][0]*sx,p["connection_mm"][1]*sy],"hidden":p["hidden"]} for p in s["pins"]]})
    if overlay_path:
        overlay_path.parent.mkdir(parents=True,exist_ok=True)
        draw=ImageDraw.Draw(image)
        for c in coords:
            if c["bbox_pixels_candidate"]:
                draw.rectangle(c["bbox_pixels_candidate"],outline=(230,120,0),width=1)
            for p in c["pins_pixels"]:
                if not p["hidden"]:
                    x,y=p["connection"];draw.ellipse((x-2,y-2,x+2,y+2),outline=(0,70,255))
        image.save(overlay_path)
    return {"status":"paper_transform_diagnostic_only", "image_size":[w,h],"scale_pixels_per_mm":[sx,sy],
            "scale_anisotropy_fraction":ratio_error,"wire_mask":"green_theme" if chosen is green else "dark_pixel_fallback",
            "wire_sample_count":len(wire_pixels),"wire_alignment_within_1px":expected,
            "shifted_13px_control_alignment_mean":shift_mean,
            "wire_alignment_support": bool(expected is not None and shift_mean is not None and expected>=.9 and expected-shift_mean>=.25 and ratio_error<.01),
            "production_label_eligible":False,"coordinates":coords,
            "caveat":"Wire registration does not independently validate symbol body bbox/pin transform, ontology, hidden geometry or dataset leakage. No YOLO labels exported."}


def write_json(path:Path,data:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")


def run(args):
    dataset=args.dataset.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    records=[json.loads(line) for line in (dataset/"manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    records=records[:args.limit] if args.limit else records
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+",str(r.get("id",""))) or r["id"] in (".","..") for r in records):
        raise ValueError("unsafe_output_record_identifier")
    groups=project_groups(records)
    inventory=Counter(p.suffix.lower() for p in dataset.rglob("*") if p.is_file())
    tally=Counter();issues=Counter();primitives=Counter();splits=Counter();rows=[];gallery=[]
    cli=shutil.which("kicad-cli")
    common=Path("C:/Program Files/KiCad")
    cli_candidates=list(common.glob("*/bin/kicad-cli.exe")) if common.exists() else []
    for index,record in enumerate(records):
        source=inside(dataset/record["source_file"],dataset);image=inside(dataset/record["image_file"],dataset)
        split=project_split(groups[record["id"]]);splits[split]+=1
        try:
            if sha256(source)!=record["source_sha256"] or sha256(image)!=record["image_sha256"]:
                raise ValueError("source_or_image_sha_mismatch")
            data=extract_source(source)
            data.update({"id":record["id"],"source_file":str(source),"source_sha256":record["source_sha256"],
                         "image_file":str(image),"image_sha256":record["image_sha256"],"dataset_revision":record["revision"],
                         "source_repository":record.get("source_repository"),"source_project_url":record.get("source_project_url"),
                         "license":record.get("dataset_license"),"project_group":groups[record["id"]],"provisional_project_split":split})
            tally["parsed_files"]+=1;tally["symbols"]+=len(data["symbols"]);tally["pins"]+=sum(len(s["pins"]) for s in data["symbols"])
            tally["symbol_bbox_candidates"]+=sum(s["bbox_mm"] is not None for s in data["symbols"])
            tally["hierarchical_sheet_links"]+=len(data["hierarchical_sheets"])
            tally["symbols_multiple_instance_paths"]+=sum(len(s["instance_paths"])>1 for s in data["symbols"])
            tally["power_symbols"]+=sum(s["source_kind"]=="power_symbol" for s in data["symbols"])
            tally["mirrored_symbols"]+=sum(bool(s["mirror"]) for s in data["symbols"])
            tally["multiunit_symbols"]+=sum(s["unit"]>1 for s in data["symbols"])
            for s in data["symbols"]:
                issues.update(s["issues"]);primitives.update(s["primitive_counts"])
            do_overlay=index<args.overlays
            if index<args.diagnostics:
                projection=diagnostic_projection(data,image,output/"overlays"/(record["id"]+".png") if do_overlay else None)
                write_json(output/"projections"/(record["id"]+".json"),projection)
                tally["diagnostic_images"]+=1;tally["wire_alignment_supported_images"]+=projection.get("wire_alignment_support",False)
                if do_overlay:
                    gallery.append((record,projection))
            write_json(output/"sources"/(record["id"]+".json"),data)
            rows.append({"id":record["id"],"status":"parsed_diagnostic_only","symbols":len(data["symbols"]),"pins":sum(len(s["pins"]) for s in data["symbols"]),"project_group":groups[record["id"]],"split":split})
        except Exception as exc:
            tally["failed_files"]+=1
            rows.append({"id":record["id"],"status":"failed","error_type":type(exc).__name__,"reason":str(exc)[:200],"project_group":groups[record["id"]],"split":split})
        if (index+1)%100==0:
            print(json.dumps({"processed":index+1,"parsed":tally["parsed_files"],"failed":tally["failed_files"]}),flush=True)
    (output/"index.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    summary={"input_records":len(records),"source_repositories":len({r.get("source_repository") for r in records}),
             "project_groups":len(set(groups.values())),"counts":dict(tally),"geometry_issues_by_symbol":dict(issues),"primitive_counts":dict(primitives),
             "provisional_split_images":dict(splits),"inventory_extensions":dict(inventory),
             "renderer":{"path_kicad_cli":cli,"program_files_kicad_cli":[str(p) for p in cli_candidates],"svg_cache_count":inventory.get(".svg",0),"renderer_launched":False},
             "production_yolo_labels":0,"training_data_modified":False,"paid_calls":0,
             "split_caveat":"Project/exact-source/exact-RGB groups kept together. Near-duplicate/fork/overlap with existing PCBnet and reviewed images NOT audited; do not use these as final splits.",
             "supported":"Modern KiCad embedded libraries, common/selected unit and body style, ref/value/footprint/property visibility/instance paths, visible/hidden pins, rectangles/polylines/circles/3-point arcs.",
             "unsupported":"Eagle/legacy SCH/XML not implemented (no such source files in this cache); external inheritance, complete typography/decorated pins, hierarchical expansion/external sheet dependencies, exact render settings, competition ontology, robust fork/near-duplicate exclusion.",
             "references":["https://dev-docs.kicad.org/en/file-formats/sexpr-schematic/","https://dev-docs.kicad.org/en/file-formats/sexpr-intro/","https://docs.kicad.org/9.0/en/cli/cli.html"]}
    write_json(output/"SUMMARY.json",summary)
    body=['<!doctype html><meta charset="utf-8"><title>Native source geometry pilot</title><style>body{font-family:Arial;max-width:1600px;margin:auto}img{max-width:100%}section{border-top:1px solid #aaa;padding:20px}pre{white-space:pre-wrap}</style><h1>原生KiCad几何候选：仅诊断，不是训练真值</h1><p>橙框：符号图形＋可见引脚线段的几何包围盒候选；蓝点：可见引脚连接点。文字不包含在橙框内。未导出YOLO标注，未混入训练。</p>']
    for record,projection in gallery:
        details={k:v for k,v in projection.items() if k!="coordinates"}
        body.append(f'<section><h2>{html.escape(record["id"])}</h2><p>{html.escape(record.get("source_repository", ""))}</p><pre>{html.escape(json.dumps(details,ensure_ascii=False,indent=2))}</pre><a href="{html.escape((dataset/record["image_file"]).resolve().as_uri())}">原图</a> | <a href="sources/{record["id"]}.json">源事实</a><img loading="lazy" src="overlays/{record["id"]}.png"></section>')
    (output/"gallery.html").write_text("\n".join(body),encoding="utf-8")
    report=f'''# 原生KiCad自动标注可行性试验\n\n本次只读取已下载数据，未运行工程脚本、未安装EDA软件、未调用付费API、未修改训练/审核数据。\n\n- 输入{len(records)}页，解析成功{tally['parsed_files']}页，失败{tally['failed_files']}页。\n- 提取{tally['symbols']}个符号、{tally['pins']}个引脚，计算{tally['symbol_bbox_candidates']}个符号几何框候选。位号、值、库ID、unit、可见性与实例路径来自源文件，不是视觉猜测。Value没有擅自当厂商型号。\n- 按{len(set(groups.values()))}个项目/精确重复组生成临时分组；未审查fork、近重复和已有数据交叉，因此还不能正式用于训练划分。\n- 对前{tally['diagnostic_images']}页按纸张尺寸投影，{tally['wire_alignment_supported_images']}页满足导线像素对齐诊断阈值。这只能验证页面变换线索，不能证明元件框和引脚全部正确。\n- 环境未发现可用KiCad CLI：PATH={bool(cli)}，常见安装目录={len(cli_candidates)}个；SVG缓存{inventory.get('.svg',0)}份。未重渲染。\n- **导出的生产YOLO标签为0，新增训练数据为0。** 所有像素框都是待校准候选。\n\n## 已支持与边界\n\n{summary['supported']}\n\n尚未解决：{summary['unsupported']}\n\n几何框约定为符号本体加可见引脚线段，不包含位号/值/引脚文字。默认线宽、引脚装饰、内部文字、非正交圆弧等存在明确issues，不能无条件当作检测真值。不同unit保留为实例，物理位号保留，不把U1A/U1B强行合并。层次子页只记录链接，不访问缓存外路径。\n\n## 接下来无需逐框人工标注的路径\n\n1. 在独立安装KiCad后，用同版本CLI重新输出SVG，解析SVG视口/页尺寸与实际符号可见图形并做回归校准，覆盖旋转/镜像/多单元/隐藏引脚。\n2. 固定本体与引脚是否包含的标注口径；以库ID+结构映射题目类别，未知库隔离，不能靠R/U前缀硬猜。\n3. 源项目、fork与图像近重复统一分组；先隔离验证/测试，再渲染多分辨率、多配色和截图。\n4. 对可证明支持的符号导出YOLO/分类/OCR关联标签；不支持部分保留在隔离区，不静默生成错误负样本。\n\n## 查看\n\n- [对齐诊断图库](gallery.html)\n- [机器统计](SUMMARY.json)\n- `sources/*.json`：每页源事实、几何候选及来源SHA\n- `projections/*.json`：像素候选与导线对齐诊断\n- `index.jsonl`：临时项目划分与成功/失败状态\n\n格式依据：[KiCad原理图格式](https://dev-docs.kicad.org/en/file-formats/sexpr-schematic/)、[通用几何/单位格式](https://dev-docs.kicad.org/en/file-formats/sexpr-intro/)、[官方CLI](https://docs.kicad.org/9.0/en/cli/cli.html)。\n'''
    (output/"REPORT.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    base=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=base.parent/"Data/open_schematics")
    parser.add_argument("--output",type=Path,default=base/"outputs/unattended_20261016/native_labels")
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--diagnostics",type=int,default=50)
    parser.add_argument("--overlays",type=int,default=8)
    run(parser.parse_args())
