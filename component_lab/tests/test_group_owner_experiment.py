import copy
import unittest

from component_lab.text_roles import parse_text_role
from component_lab.text_matching import associate_structured
from component_lab.contest_components import associate_variants, export_contest
from component_lab.group_owner_experiment import ReverseGroupOwner, clear_by_three_locks


def fixture(parameter='10k'):
    boxes=[{'id':'left','type':'resistor','bbox':[0,30,10,50]},
           {'id':'right','type':'resistor','bbox':[35,30,45,50]}]
    texts=[{'id':'ref','raw':'R1','bbox':[17.5,15,27.5,25]},
           {'id':'val','raw':parameter,'bbox':[0,30,20,40]}]
    return boxes,texts


class GroupOwnerTests(unittest.TestCase):
    def test_numeric_units_are_opt_in_and_raw_preserved(self):
        for raw in ['U2.2','IC12.23']:
            row={'raw':raw,'bbox':[0,0,30,10]}
            self.assertFalse(any(r['role']=='reference' for r in parse_text_role(row)['role_candidates']))
            parsed=parse_text_role(row,numeric_unit_refs=True)
            ref=next(r for r in parsed['role_candidates'] if r['role']=='reference')
            self.assertEqual(ref['value'],raw)
            self.assertEqual(parsed['raw'],raw)
            self.assertEqual(ref['physical_reference'],raw.split('.')[0])
        for raw in ['USB1','SG90','V0.9','R1.2']:
            self.assertFalse(any(r['role']=='reference' for r in parse_text_role({'raw':raw},numeric_unit_refs=True)['role_candidates']))

    def test_numeric_unit_prior_and_export(self):
        boxes=[{'id':'u','type':'component','bbox':[0,20,20,40]}]
        texts=[{'id':'r','raw':'u2.2','bbox':[0,0,20,10]}]
        result,_=associate_variants(boxes,texts,numeric_unit_refs=True)
        pred,_=export_contest(result['identity_v1'],60)
        self.assertEqual(pred['components']['u2.2']['type'],'box')

    def test_reverse_parameter_breaks_reference_tie(self):
        boxes,texts=fixture(); before=copy.deepcopy((boxes,texts))
        base=associate_structured(boxes,texts,clamp_costs=False)
        self.assertTrue(all(c['reference'] is None for c in base['components']))
        adjuster=ReverseGroupOwner()
        out=associate_structured(boxes,texts,clamp_costs=False,reference_cost_adjuster=adjuster)
        self.assertEqual(out['components'][0]['reference'],'R1')
        self.assertEqual(out['components'][1]['reference'],None)
        self.assertEqual((boxes,texts),before)
        self.assertGreater(adjuster.counts['adjusted_reference_rows'],0)

    def test_numeric_context_cannot_be_independent_anchor(self):
        boxes,texts=fixture('5100'); adjuster=ReverseGroupOwner()
        out=associate_structured(boxes,texts,clamp_costs=False,reference_cost_adjuster=adjuster)
        self.assertTrue(all(c['reference'] is None for c in out['components']))
        self.assertEqual(adjuster.counts['adjusted_reference_rows'],0)

    def test_parameter_symmetry_retains_abstention(self):
        boxes,texts=fixture();texts[1]['bbox']=[12.5,30,32.5,40]
        adjuster=ReverseGroupOwner()
        out=associate_structured(boxes,texts,clamp_costs=False,reference_cost_adjuster=adjuster)
        self.assertTrue(all(c['reference'] is None for c in out['components']))

    def test_reverse_is_scale_invariant(self):
        boxes,texts=fixture()
        small=associate_structured(boxes,texts,clamp_costs=False,reference_cost_adjuster=ReverseGroupOwner())
        for row in boxes+texts:row['bbox']=[v*4 for v in row['bbox']]
        large=associate_structured(boxes,texts,clamp_costs=False,reference_cost_adjuster=ReverseGroupOwner())
        self.assertEqual([(a['component_id'],a['text_id']) for a in small['assignments']],
                         [(a['component_id'],a['text_id']) for a in large['assignments']])

    def test_no_group_is_identical_and_does_not_expand_candidates(self):
        boxes,texts=fixture();texts=texts[:1]
        base=associate_structured(boxes,texts,clamp_costs=False)
        out=associate_structured(boxes,texts,clamp_costs=False,reference_cost_adjuster=ReverseGroupOwner())
        self.assertEqual(base,out)

    def test_clear_guard_is_explicit_copy_not_null_truth(self):
        boxes=[{'id':'r','bbox':[0,30,20,60],'type':'resistor'}]
        texts=[{'id':'r1','raw':'R1','bbox':[0,0,20,10]}, {'id':'v','raw':'10k','bbox':[22,40,42,50]}]
        a=associate_structured(boxes,texts,clamp_costs=False)
        pred,trace=export_contest(a,100);pred['components']['R1']['value']='10k'
        before=copy.deepcopy((a,pred,trace))
        out,decisions=clear_by_three_locks(a,pred,trace)
        self.assertIsNone(out['components']['R1']['value'])
        self.assertTrue(decisions[0]['cleared'])
        self.assertEqual((a,pred,trace),before)

    def test_guard_keeps_tight_reference_value_group(self):
        boxes=[{'id':'r','bbox':[30,30,50,60],'type':'resistor'}]
        texts=[{'id':'r1','raw':'R1','bbox':[55,32,75,42]}, {'id':'v','raw':'10k','bbox':[55,48,80,58]}]
        a=associate_structured(boxes,texts,clamp_costs=False)
        pred,trace=export_contest(a,100)
        out,_=clear_by_three_locks(a,pred,trace)
        self.assertEqual(out['components']['R1']['value'],'10k')
