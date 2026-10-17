import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image
from component_lab.pipeline import infer


class IdentityPipelineTests(unittest.TestCase):
    def experimental_infer(self, texts, kind="component", **flags):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); image=root/'image.png';Image.new('RGB',(120,120),'white').save(image)
            detections,ocr=root/'d.json',root/'o.json'
            detections.write_text(json.dumps({'components':[{'id':'a','type':kind,'bbox':[20,30,100,100]}]}),'utf-8')
            ocr.write_text(json.dumps({'texts':texts}),'utf-8')
            args=SimpleNamespace(image=str(image),detections=str(detections),ocr_json=str(ocr),output=str(root/'out'),
                                 resnet=None,qwen=False,qwen_max_calls=0,context_margin=20,text_margin=60,
                                 classifier_threshold=.8,association='identity-v1',contest_export=True,**flags)
            infer(args)
            return json.loads((root/'out/contest_components.json').read_text('utf-8'))

    def test_optional_display_name_cli_path(self):
        p=self.experimental_infer([{'text':'U1','bbox':[20,10,35,20]},
                                   {'text':'oled','bbox':[20,21,70,29]}],kind='ic_or_module',display_names=True)
        self.assertEqual(p['components']['U1']['Name'],'oled')

    def test_optional_compound_span_cli_path(self):
        p=self.experimental_infer([{'text':'R1 10kΩ','bbox':[20,10,90,20]}],ocr_spans=True)
        self.assertIn('R1',p['components'])
        self.assertEqual(p['components']['R1']['type'],'r')
    def test_cached_models_to_component_contest_export_without_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (100, 100), "white").save(image)
            detections, ocr = root / "detections.json", root / "ocr.json"
            detections.write_text(json.dumps({"components": [{"id": "a", "type": "component", "bbox": [20, 20, 40, 50]}]}), "utf-8")
            ocr.write_text(json.dumps({"texts": [{"text": "R1", "bbox": [45, 20, 65, 30]},
                                                       {"text": "10kΩ", "bbox": [45, 35, 70, 45]}]}), "utf-8")
            args = SimpleNamespace(image=str(image), detections=str(detections), ocr_json=str(ocr),
                                   output=str(root / "out"), resnet=None, qwen=False, qwen_max_calls=0,
                                   context_margin=20, text_margin=60, classifier_threshold=.8,
                                   association="identity-v1", contest_export=True)
            result = infer(args)
            self.assertEqual(result["components"], 1)
            exported = json.loads((root / "out/contest_components.json").read_text("utf-8"))
            self.assertEqual(exported["components"]["R1"], {"type": "r", "bbox": [20, 50, 40, 80], "Name": None, "value": "10kΩ"})
            self.assertEqual(exported["pins"], {})
            self.assertEqual(exported["nets"], {})
