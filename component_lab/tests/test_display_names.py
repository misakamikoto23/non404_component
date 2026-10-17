import copy
import unittest
from component_lab.display_names import assign_display_names
from component_lab.contest_components import export_contest


def fixture(raw='oled',role='description'):
    return {'components':[{'id':'u','type':'ic_or_module','bbox':[20,30,100,100],
                           'association_evidence':{'reference':{'raw':'U1','text_id':'ref'}}}],
            'texts':[{'id':'ref','raw':'U1','bbox':[20,10,35,20],'role_candidates':[{'role':'reference'}]},
                     {'id':'name','raw':raw,'bbox':[20,21,70,29],'role_candidates':[{'role':role}]}]}


class DisplayNameTests(unittest.TestCase):
    def test_literal_names_and_no_mutation(self):
        for raw in ['oled','N20电机','{ManufacturerPart!}']:
            a=fixture(raw);before=copy.deepcopy(a);out=assign_display_names(a)
            self.assertEqual(a,before)
            pred,_=export_contest(out,120,display_names=True)
            self.assertEqual(pred['components']['U1']['Name'],raw)

    def test_net_and_pin_tokens_excluded(self):
        for role in ['net','pin','reference','value']:
            out=assign_display_names(fixture('GPIO0',role))
            self.assertEqual(out['display_name_diagnostics']['assigned'],0)

    def test_existing_part_number_not_overwritten(self):
        a=fixture();a['components'][0]['association_evidence']['part_number']={'raw':'LM358','text_id':'other'}
        out=assign_display_names(a);self.assertEqual(out['display_name_diagnostics']['assigned'],0)

    def test_same_reference_not_copied_as_name(self):
        a=fixture();a['texts']=a['texts'][:1]
        out=assign_display_names(a);pred,_=export_contest(out,120,display_names=True)
        self.assertIsNone(pred['components']['U1']['Name'])
