import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
import native_geometry_calibration as calibration
import native_schematic_label_pilot as native


class CalibrationTests(unittest.TestCase):
    def test_explicit_wire_color_alpha_and_antialiasing(self):
        self.assertEqual(calibration.wire_color(native.parse_sexpr('(wire (stroke (color 194 0 0 1)))')),(194,0,0))
        self.assertEqual(calibration.wire_color(native.parse_sexpr('(wire (stroke (color 0 0 0 0)))')),"default_green")
        image=np.full((20,20,3),245,dtype=np.uint8)
        image[10,10]=[194,0,0];image[10,11]=[220,122,122]
        mask=calibration.color_mask(image,(194,0,0))
        self.assertTrue(mask[10,10]);self.assertTrue(mask[10,11]);self.assertFalse(mask[0,0])

    def test_wire_segment_partition_is_disjoint(self):
        root=native.parse_sexpr('(root (wire (pts (xy 1 2)(xy 8 2)))(wire (pts (xy 2 3)(xy 9 3))))')
        fit,heldout,counts,endpoints=calibration.wire_points(root,10,10)
        self.assertEqual(counts["fit_segments"]+counts["heldout_segments"],2)
        a={tuple(p) for points in fit.values() for p in points}
        b={tuple(p) for points in heldout.values() for p in points}
        self.assertFalse(a&b)
        self.assertEqual(len(endpoints),4)

    def test_registration_matches_source_color_and_rejects_shift(self):
        image=Image.new("RGB",(400,400),"white");draw=ImageDraw.Draw(image)
        wires=[]
        for y in range(2,35,2):
            wires.append(f'(wire (pts (xy 2 {y})(xy 35 {y})) (stroke (color 194 0 0 1)))')
            draw.line((20,y*10,350,y*10),fill=(194,0,0),width=1)
        root=native.parse_sexpr('(root '+''.join(wires)+')')
        good,_,_,_=calibration.registration(root,np.asarray(image),[40,40])
        self.assertTrue(good["registration_supported"])
        self.assertFalse(good["production_label_eligible"])
        shifted=Image.new("RGB",(400,400),"white");draw=ImageDraw.Draw(shifted)
        for y in range(2,35,2):
            draw.line((20,y*10+7,350,y*10+7),fill=(194,0,0),width=1)
        bad,_,_,_=calibration.registration(root,np.asarray(shifted),[40,40])
        self.assertFalse(bad["registration_supported"])

    def test_dnp_state_preserved_on_pin_extraction(self):
        root=native.parse_sexpr('''(root (lib_symbols (symbol "Test:R"
           (symbol "R_1_1" (pin passive line (at 0 0 0) (length 2) (name "A") (number "1")))))
           (symbol (lib_id "Test:R") (at 10 10 0) (unit 1) (dnp yes)))''')
        pins=list(calibration.raw_visible_pins(root))
        self.assertEqual(len(pins),1);self.assertTrue(pins[0]["dnp"])


if __name__=="__main__":
    unittest.main()
