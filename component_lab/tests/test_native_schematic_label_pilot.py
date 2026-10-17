"""Tests for source-traceable extraction; these do not certify renderer calibration."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/native_schematic_label_pilot.py"
SPEC = importlib.util.spec_from_file_location("native_pilot", MODULE_PATH)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class NativeSourceTests(unittest.TestCase):
    def test_quoted_properties_and_invalid_input(self):
        self.assertEqual(native.parse_sexpr('(root (property "Value" "R \\"quoted\\"\\n10k"))')[1][2], 'R "quoted"\n10k')
        for bad in ('(root', '(root))', '(root) (other)', '(root "unterminated)'):
            with self.assertRaises(ValueError):
                native.parse_sexpr(bad)
        with self.assertRaises(ValueError):
            native.parse_sexpr('((((x))))', max_depth=2)

    def test_coordinate_and_mirror_convention(self):
        self.assertEqual(native.transform_point((2,3),[10,20,0]), (12,17))
        self.assertEqual(native.transform_point((2,3),[10,20,0],"x"), (12,23))
        x,y=native.transform_point((2,3),[10,20,90])
        self.assertAlmostEqual(x,7);self.assertAlmostEqual(y,18)

    def test_common_unit_selection_hidden_pin_and_properties(self):
        source='''(kicad_sch (version 20231120) (uuid "page") (paper "A4")
          (lib_symbols (symbol "Test:Dual"
            (symbol "Dual_0_1" (rectangle (start -2 -1) (end 2 1) (stroke (width 0.2))))
            (symbol "Dual_1_1" (pin input line (at -5 0 0) (length 3) (name "IN") (number "1")))
            (symbol "Dual_2_1" (pin output line (at 6 0 180) (length 4) (name "OUT") (number "2")))
            (symbol "Dual_0_0" (pin power_in line (at 0 8 270) (length 2) hide (name "VCC") (number "8")))))
          (symbol (lib_id "Test:Dual") (at 20 30 0) (unit 1) (uuid "symbol")
            (property "Reference" "U1" (at 20 25 0))
            (property "Value" "LM358" (at 20 35 0) (effects (hide yes)))
            (instances (project "test" (path "/page" (reference "U1") (unit 1))))))'''
        path=Path("fixture.kicad_sch")
        with patch.object(Path,"read_text",return_value=source), patch.object(Path,"stat",return_value=SimpleNamespace(st_size=len(source))):
            data=native.extract_source(path)
        s=data["symbols"][0]
        self.assertEqual(s["reference"],"U1");self.assertEqual(s["value"],"LM358")
        self.assertIsNone(s["part_number"])
        self.assertEqual({p["number"] for p in s["pins"]},{"1","8"})
        self.assertEqual(s["pins"][0]["connection_mm"],[15,30])
        self.assertTrue(s["pins"][1]["hidden"])
        self.assertAlmostEqual(s["bbox_mm"][0],14.9)
        self.assertAlmostEqual(s["bbox_mm"][1],28.9)
        self.assertTrue(s["properties"]["Value"]["hidden"])
        self.assertEqual(s["instance_paths"][0]["sheet_path"],"/page")

    def test_arc_extremum_not_just_start_and_end(self):
        node=native.parse_sexpr('(arc (start 1 0) (mid 0 1) (end -1 0))')
        pts,exact=native.arc_points(node)
        self.assertTrue(exact)
        self.assertAlmostEqual(max(y for x,y in pts),1)
        self.assertGreaterEqual(min(y for x,y in pts),-1e-12)

    def test_project_and_duplicate_group_separation(self):
        rows=[{"id":"a","source_repository":"Owner/Repo","source_sha256":"s1","rgb_sha256":"i1"},
              {"id":"b","source_repository":"owner/repo","source_sha256":"s2","rgb_sha256":"i2"},
              {"id":"c","source_repository":"Fork/Repo","source_sha256":"s2","rgb_sha256":"i3"},
              {"id":"d","source_repository":"Other/Repo","source_sha256":"s4","rgb_sha256":"i4"}]
        groups=native.project_groups(rows)
        self.assertEqual(groups["a"],groups["b"]);self.assertEqual(groups["b"],groups["c"])
        self.assertNotEqual(groups["a"],groups["d"])
        self.assertEqual(native.project_split(groups["a"]),native.project_split(groups["c"]))

    def test_unresolved_inheritance_is_quarantined(self):
        lib=native.parse_sexpr('(symbol "X" (extends "Y"))')
        geometry=native.symbol_geometry(lib,1,1,[0,0,0],"")
        self.assertIsNone(geometry["bbox_mm"])
        self.assertIn("inherited_library_not_resolved",geometry["issues"])

    def test_portrait_and_cache_path_guard(self):
        self.assertEqual(native.paper_size(native.parse_sexpr('(root (paper "A4" portrait))')),(210,297))
        root=MODULE_PATH.parent.resolve()
        self.assertEqual(native.inside(root/"file",root),root/"file")
        with self.assertRaises(ValueError):
            native.inside(root/".."/"outside",root)

    def test_wire_registration_is_diagnostic_and_rejects_misalignment(self):
        from PIL import Image, ImageDraw
        data={"paper_mm":[10,10],"wires_mm":[[[1,5],[9,5]]],"symbols":[]}
        aligned=Image.new("RGB",(100,100),"white")
        ImageDraw.Draw(aligned).line((10,50,90,50),fill=(0,128,0),width=1)
        with patch("PIL.Image.open",return_value=aligned):
            result=native.diagnostic_projection(data,Path("unused.png"))
        self.assertTrue(result["wire_alignment_support"])
        self.assertFalse(result["production_label_eligible"])
        shifted=Image.new("RGB",(100,100),"white")
        ImageDraw.Draw(shifted).line((10,80,90,80),fill=(0,128,0),width=1)
        with patch("PIL.Image.open",return_value=shifted):
            result=native.diagnostic_projection(data,Path("unused.png"))
        self.assertFalse(result["wire_alignment_support"])


if __name__=="__main__":
    unittest.main()
