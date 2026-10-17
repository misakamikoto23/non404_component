import unittest
from PIL import Image
from component_lab.qwen_api import build_messages, parse_response


class QwenTests(unittest.TestCase):
    def test_two_image_payload(self):
        image = Image.new("RGB", (10, 10), "white")
        messages = build_messages(image, image, {"id": "c1", "type": "r", "bbox": [0, 0, 10, 10]})
        content = messages[0]["content"]
        self.assertEqual(len(content), 3)
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_json_response(self):
        data = parse_response('```json\n{"type":"r","reference":"R1","Name":null,"value":"10k","texts":["R1"],"uncertain":false}\n```')
        self.assertEqual(data["reference"], "R1")

    def test_invalid_response_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_response('{"texts":"R1","uncertain":false}')
