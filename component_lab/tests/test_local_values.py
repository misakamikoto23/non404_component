import copy
import unittest
from component_lab.local_values import fill_local_values,pair_cost
from component_lab.text_roles import parse_texts


def fixture():
    texts=parse_texts([{'id':'r','raw':'R1','bbox':[0,0,20,10]},
                      {'id':'v','raw':'4.7K','bbox':[0,14,25,24]}])
    return {'texts':texts,'components':[{'id':'a','type':'resistor',
            'association_evidence':{'reference':{'raw':'R1','text_id':'r'}}}]}


class LocalValuesTests(unittest.TestCase):
    def test_decimal_upper_K_is_literal_value(self):
        a=fixture();self.assertIn('value',[x['role'] for x in a['texts'][1]['role_candidates']])
        before=copy.deepcopy(a);r=fill_local_values(a)
        self.assertEqual(a,before)
        self.assertEqual(r['components'][0]['association_evidence']['value']['raw'],'4.7K')

    def test_existing_owner_value_is_never_overwritten(self):
        a=fixture();a['components'][0]['association_evidence']['value']={'raw':'10k','text_id':'other'}
        self.assertEqual(fill_local_values(a)['components'][0]['association_evidence']['value']['raw'],'10k')

    def test_other_owner_token_locked(self):
        a=fixture();a['components'].append({'id':'b','type':'ic','association_evidence':{'part_number':{'text_id':'v','raw':'4.7K'}}})
        self.assertEqual(fill_local_values(a)['local_value_diagnostics']['filled'],0)

    def test_reference_between_blocks_pair(self):
        ref={'id':'r','bbox':[0,0,20,10]};value={'id':'v','bbox':[0,22,20,32]}
        other={'id':'r2','bbox':[0,12,20,22]}
        self.assertIsNone(pair_cost(ref,value,[ref,other]))

    def test_orientation_difference_abstains(self):
        ref={'id':'r','bbox':[0,0,20,10],'orientation':0}
        value={'id':'v','bbox':[0,14,20,24],'orientation':90}
        self.assertIsNone(pair_cost(ref,value,[ref]))

    def test_duplicate_references_abstain(self):
        a=fixture();a['components'].append({'id':'b','type':'resistor','association_evidence':{'reference':{'raw':'R1','text_id':'r'}}})
        self.assertEqual(fill_local_values(a)['local_value_diagnostics']['filled'],0)
