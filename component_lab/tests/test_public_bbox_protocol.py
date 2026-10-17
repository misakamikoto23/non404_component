import unittest
from scripts.probe_public_bbox_protocol import adapt_prediction


class ProtocolBBoxTests(unittest.TestCase):
    def test_only_explicit_types_center_preserved_and_input_immutable(self):
        p={'components':{'R1':{'type':'r','bbox':[10,20,70,40]},'U1':{'type':'box','bbox':[0,0,100,100]}},'pins':{},'nets':{}}
        q,trace=adapt_prediction(p,{'r':{'short':8,'long':20}})
        self.assertEqual(q['components']['R1']['bbox'],[30,26,50,34])
        self.assertEqual(p['components']['R1']['bbox'],[10,20,70,40])
        self.assertEqual(q['components']['U1'],p['components']['U1'])
        self.assertEqual(len(trace),1)

    def test_vertical_from_prediction_not_target(self):
        p={'components':{'C1':{'type':'c','bbox':[0,0,10,60]}},'pins':{},'nets':{}}
        q,_=adapt_prediction(p,{'c':{'short':16,'long':20}})
        self.assertEqual(q['components']['C1']['bbox'],[-3,20,13,40])
