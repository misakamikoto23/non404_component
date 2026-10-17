from copy import deepcopy
import unittest

from component_lab.recovery_gate import recovery_candidates, sample_candidates


class RecoveryGateTests(unittest.TestCase):
    def fixture(self):
        component = {'id':'c1','bbox':[20,20,30,40],'type':'resistor',
                     'detection':{'score':.9},'classification':{'score':.9}}
        text = {'id':'t1','raw':'R2','bbox':[32,20,42,30],'score':.99,
                'role_candidates':[{'role':'reference','heuristic_score':.9,'prefix':'R'}]}
        association = {'components':[component],'texts':[text],'assignments':[]}
        traces = [{'source_id':'c1','key':'__unresolved_detection_0001'}]
        baseline = {'components':{'__unresolved_detection_0001':{'type':'r'}}}
        return baseline,association,traces

    def gate(self, fixture):
        return recovery_candidates('a',*fixture,100,100)

    def test_only_unresolved_and_unlocked_reference_is_admitted(self):
        fixture=self.fixture();rows,_=self.gate(fixture)
        self.assertEqual(rows[0]['tier'],'reference_candidate')
        baseline,a,t=fixture;baseline['components']['R2']={'type':'r'}
        rows,rejected=self.gate(fixture);self.assertEqual(rows,[])
        self.assertEqual(rejected,{'no_text_group_evidence':1})
        t[0]['key']='R1'
        rows,_=self.gate(fixture);self.assertEqual(rows,[])

    def test_ports_low_scores_and_frame_do_not_enter_paid_pool(self):
        for extra in [{'type':'ground'},{'classification':{'score':.5}},
                      {'detection':{'score':.2}},{'bbox':[0,0,90,90]}]:
            fixture=self.fixture();fixture[1]['components'][0].update(extra)
            rows,_=self.gate(fixture)
            self.assertEqual(rows,[])

    def test_owner_and_reference_syntax_are_checked(self):
        fixture=self.fixture();fixture[1]['assignments']=[{'text_id':'t1','component_id':'neighbor','role':'reference'}]
        self.assertEqual(self.gate(fixture)[0],[])
        fixture=self.fixture();fixture[1]['texts'][0]['raw']='CIN'
        self.assertEqual(self.gate(fixture)[0],[])

    def test_dense_pin_number_rows_are_not_demoted_to_field_tier(self):
        baseline,a,t=self.fixture();a['components'][0]['type']='connector'
        template=a['texts'][0]
        a['texts']=[dict(template,id=f'p{i}',raw=f'P{i}',role_candidates=[{'role':'reference','prefix':'P','heuristic_score':.9}]) for i in range(10)]
        rows,rejected=self.gate((baseline,a,t))
        self.assertEqual(rows,[]);self.assertEqual(rejected,{'dense_pin_number_row':1})

    def test_context_dependent_numeric_is_not_independent_field_anchor(self):
        fixture=self.fixture();text=fixture[1]['texts'][0]
        text.update(raw='100',role_candidates=[{'role':'value','value_kind':'resistance','requires_context':True,'heuristic_score':.9}])
        self.assertEqual(self.gate(fixture)[0],[])
        text.update(raw='100Ω',role_candidates=[{'role':'value','value_kind':'resistance','heuristic_score':.9}])
        self.assertEqual(self.gate(fixture)[0][0]['tier'],'field_context_only')

    def test_sampling_is_order_independent_and_respects_page_cap_and_exclusion(self):
        rows=[{'case':case,'source_id':f'{case}{tier}{i}','tier':tier,'selection_hash':str(i)}
              for case in ['a','b','c'] for tier in ['reference_candidate','field_context_only'] for i in range(3)]
        exclude={('a','areference_candidate0')}
        selected=sample_candidates(rows,per_tier=3,per_case=2,exclude_sources=exclude)
        self.assertEqual(selected,sample_candidates(list(reversed(rows)),per_tier=3,per_case=2,exclude_sources=exclude))
        self.assertEqual(len(selected),6)
        self.assertNotIn(('a','areference_candidate0'),{(r['case'],r['source_id']) for r in selected})
        for case in ['a','b','c']:
            self.assertLessEqual(sum(r['case']==case for r in selected),2)


if __name__=='__main__':unittest.main()
