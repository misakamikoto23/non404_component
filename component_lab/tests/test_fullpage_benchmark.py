"""Geometry, annotation isolation, billing and resume without paid API calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_qwen_fullpage as b


class BenchmarkTests(unittest.TestCase):
    def test_normalized_coordinates_and_invalid_boxes(self):
        preds, invalid = b.parse_boxes(json.dumps({'boxes': [[100,200,300,400],[0,0,0,2],[-1,0,2,2],[True,0,2,2]]}), 2000, 500)
        self.assertEqual(preds[0]['bbox'], [200,100,600,200])
        self.assertEqual(invalid, [1,2,3])
        with self.assertRaises(ValueError):
            b.parse_boxes('{"components":[]}',100,100)
        with self.assertRaises(ValueError):
            b.parse_boxes('{"boxes":[[0,0,100,100]],"labels":["R1"]}',100,100)
        recovered, _ = b.parse_boxes('{"boxes":[[0,0,100,100]],"labels":["R1"]}',100,100,allow_extra_keys=True)
        self.assertEqual(recovered[0]['bbox'],[0,0,10,10])

    def test_payload_is_annotation_blind(self):
        config = {'prompt':'test','max_tokens':10,'temperature':0,'enable_thinking':False,'ground_truth':'PRIVATE_GT_SENTINEL'}
        request = b.payload('model',b'abc',config)
        self.assertNotIn('PRIVATE_GT_SENTINEL',json.dumps(request))
        self.assertEqual(len(request['messages'][0]['content']),2)
        self.assertTrue(request['messages'][0]['content'][1]['image_url']['url'].endswith('YWJj'))

    def test_duplicate_and_invalid_boxes_are_false_positives(self):
        gt = [{'bbox':[0,0,10,10],'class_id':0}]
        pred = {'bbox':[0,0,10,10],'class_id':0,'score':1}
        stats, _ = b.evaluate(gt,[pred,pred],.5,invalid=1)
        self.assertEqual((stats['tp'],stats['fp'],stats['fn']), (1,2,0))

    def test_call_accounting_and_failed_resume_no_retry(self):
        import requests
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            out = Path(folder)
            (out/'inputs').mkdir()
            image = out/'inputs/a.png'
            image.write_bytes(b'fake-png')
            selection = {'samples':[{'case':'a','image':'inputs/a.png','api_sha256':b.sha(image),
                         'api_size':[100,100],'width':100,'height':100}],
                         'prompt':'test','temperature':0,'max_tokens':16000,'enable_thinking':False}
            b.atomic_json(out/'selection.json',selection)
            response = unittest.mock.Mock(status_code=200)
            response.json.return_value = {'model':'returned','usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120},
                'choices':[{'finish_reason':'stop','message':{'content':'{"boxes":[[100,100,200,200]]}'}}]}
            with patch.object(b,'credential',return_value='secret'), patch.object(requests.Session,'post',return_value=response) as post, patch.object(b,'report'):
                b.run(out,None)
                self.assertEqual(post.call_count,2)
                b.run(out,None)
                self.assertEqual(post.call_count,2)
            rows = b.read(out/'ledger.json')['requests']
            self.assertTrue(all(r['status']=='ok' and r['accounted_tokens']==120 for r in rows))
            rows[0]['status']='in_flight'
            b.atomic_json(out/'ledger.json',{'requests':rows,'token_budget':450000})
            with patch.object(b,'credential',return_value='secret'), patch.object(requests.Session,'post') as post, patch.object(b,'report'):
                b.run(out,None)
                post.assert_not_called()

if __name__ == '__main__':
    unittest.main()
