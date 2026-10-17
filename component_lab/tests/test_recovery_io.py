from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from component_lab.cli import main
from component_lab.identity_recovery import record_fingerprint
from component_lab.recovery_io import recover_files


class RecoveryIoTests(unittest.TestCase):
    def fixture(self, root):
        pred = {'a':{'components':{'R1':{'type':'r','bbox':[0,0,10,10],'Name':None,'value':None},
                '__unresolved_detection_0001':{'type':'component','bbox':[20,0,30,10],'Name':None,'value':None}},'pins':{},'nets':{}}}
        targets = [{'id':'T1','case':'a','baseline_key':'__unresolved_detection_0001',
                    'expected_record_sha256':record_fingerprint(pred['a']['components']['__unresolved_detection_0001'])}]
        manifest = {'baseline_sha256':record_fingerprint(pred),'targets':targets}
        readings = {'items':[{'id':'T1','reference':'R2','Name':None,'value':'10 kΩ',
                             'family':'resistor','target_complete':True,'uncertain_fields':[]}]}
        files=[]
        for name,data in [('baseline',pred),('manifest',manifest),('readings',readings)]:
            path=root/f'{name}.json';path.write_text(json.dumps(data),encoding='utf-8');files.append(path)
        return files,pred,manifest,readings

    def test_real_cli_defaults_to_key_only_without_any_api(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);files,pred,_,_=self.fixture(root);out=root/'result'
            argv=['component-lab','recover-identities','--baseline',str(files[0]),'--manifest',str(files[1]),
                  '--readings',str(files[2]),'--output',str(out)]
            with patch('sys.argv',argv),redirect_stdout(io.StringIO()) as stdout:
                main()
            report=json.loads(stdout.getvalue());self.assertEqual(report['api_calls'],0)
            self.assertFalse(report['with_fields'])
            actual=json.loads((out/'a.json').read_text(encoding='utf-8'))
            self.assertIsNone(actual['components']['R2']['value'])
            self.assertEqual(actual['components']['R1'],pred['a']['components']['R1'])

    def test_explicit_fields_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);files,_,_,_=self.fixture(root);out=root/'result'
            recover_files(*files,out,with_fields=True)
            self.assertEqual(json.loads((out/'a.json').read_text(encoding='utf-8'))['components']['R2']['value'],'10 kΩ')
            with self.assertRaises(FileExistsError):recover_files(*files,out)

    def test_stale_baseline_and_unknown_response_target_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);files,pred,_,readings=self.fixture(root);out=root/'result'
            pred['a']['components']['R1']['value']='modified'
            files[0].write_text(json.dumps(pred),encoding='utf-8')
            with self.assertRaises(ValueError):recover_files(*files,out)
            self.assertFalse(out.exists())
            files,_,_,readings=self.fixture(root);readings['items'][0]['id']='UNKNOWN'
            files[2].write_text(json.dumps(readings),encoding='utf-8')
            with self.assertRaises(ValueError):recover_files(*files,out)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
