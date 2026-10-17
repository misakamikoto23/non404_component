from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pilot_round005_safe_fusion import fuse


def component(kind='component',bbox=None):
    return dict(type=kind,bbox=bbox or [0,0,10,10],Name=None,value='ORIGINAL')


def bundle(parts):
    return dict(components=parts,pins={'keep':1},nets={'keep':2})


class SafeFusionTests(unittest.TestCase):
    def test_recovery_keeps_bbox_fields_count_and_old_named(self):
        base=bundle({'R1':component('r',[50,0,60,10]),'__unresolved_detection_1':component()})
        proposal=component('r',[.1,0,10,10]);proposal.update(Name='CHANGED',value='CHANGED')
        before=copy.deepcopy(base)
        pred,log=fuse(base,bundle({'R2':proposal}))
        self.assertEqual(base,before)
        self.assertEqual(pred['components']['R1'],base['components']['R1'])
        self.assertEqual(pred['components']['R2']['bbox'],[0,0,10,10])
        self.assertEqual(pred['components']['R2']['value'],'ORIGINAL')
        self.assertIsNone(pred['components']['R2']['Name'])
        self.assertEqual(len(pred['components']),2)
        self.assertEqual(log['old_named_hash'],log['output_named_hash'])

    def test_existing_key_is_never_overwritten(self):
        base=bundle({'R1':component('r'),'__unresolved_detection_1':component()})
        pred,log=fuse(base,bundle({'R1':component('c')}))
        self.assertEqual(pred,base)
        self.assertEqual(log['changes'],[])

    def test_ambiguous_new_keys_are_rejected(self):
        base=bundle({'__unresolved_detection_1':component()})
        pred,log=fuse(base,bundle({'R2':component('r'),'R3':component('r')}))
        self.assertEqual(pred,base)
        self.assertFalse(log['changes'])

    def test_incompatible_existing_family_is_rejected(self):
        base=bundle({'__unresolved_detection_1':component('c')})
        pred,log=fuse(base,bundle({'R2':component('r')}))
        self.assertEqual(pred,base)


if __name__=='__main__':
    unittest.main()
