import copy
import json
import unittest

from scripts.pilot_qwen_direct_text import apply_direct, parse_direct


class DirectTextTests(unittest.TestCase):
    def fixture(self):
        b={'a':{'components':{'R1':{'type':'r','bbox':[0,0,20,10],'Name':None,'value':None}},'pins':{},'nets':{}}}
        t=[{'id':'T1','case':'a','baseline_key':'R1'}]
        r=[{'id':'T1','reference':'R9','Name':None,'value':'10 kΩ','uncertain_fields':['Name']}]
        return b,t,r

    def test_named_key_stays_and_raw_value_preserved(self):
        b,t,r=self.fixture();before=copy.deepcopy(b)
        out,meta=apply_direct(b,t,parse_direct(json.dumps({'items':r}),t),True)
        self.assertEqual(b,before);self.assertEqual(list(out['a']['components']),['R1'])
        self.assertEqual(out['a']['components']['R1']['value'],'10 kΩ')
        self.assertEqual(meta['changes'][0]['source'],'qwen_visual_literal_not_verified')

    def test_uncertain_and_nonempty_locked(self):
        b,t,r=self.fixture();b['a']['components']['R1']['value']='10K'
        out,_=apply_direct(b,t,r,True);self.assertEqual(out,b)
        r[0]['uncertain_fields'].append('value')
        out,_=apply_direct(b,t,r,False);self.assertEqual(out,b)

    def test_invalid_schemas(self):
        _,t,r=self.fixture()
        for bad in [{'items':[]},{'items':r+r},{'items':[dict(r[0],value=12)]},
                    {'items':[dict(r[0],uncertain_fields=['bbox'])]}]:
            with self.assertRaises(ValueError):parse_direct(json.dumps(bad),t)

    def test_duplicate_new_reference_rejected(self):
        b,t,r=self.fixture()
        item=b['a']['components'].pop('R1')
        b['a']['components']={'__unresolved_detection_1':copy.deepcopy(item),'__unresolved_detection_2':copy.deepcopy(item)}
        t[0]['baseline_key']='__unresolved_detection_1';t.append(dict(t[0],id='T2',baseline_key='__unresolved_detection_2'))
        r.append(dict(r[0],id='T2'))
        out,meta=apply_direct(b,t,r,True)
        self.assertNotIn('R9',out['a']['components']);self.assertEqual(len(meta['blocked']),2)
