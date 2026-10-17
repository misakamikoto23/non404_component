"""Model-free checks for the controlled architecture experiment."""
from collections import Counter
import importlib.util
from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from compare_component_architectures import balanced_indices,context_box
from evaluate_component_architectures import export_shared,hybrid_choice,pair_counts


def test_balanced_exposures_without_inventing_unseen_class():
    targets=[0,0,0,3]
    indices=balanced_indices(targets,epoch=2)
    assert Counter(targets[i] for i in indices)=={0:64,3:64}
    assert indices==balanced_indices(targets,epoch=2)
    assert indices!=balanced_indices(targets,epoch=3)
    assert set(indices)<=set(range(len(targets)))


def test_context_crop_clips_identically_for_training_and_predictions():
    assert context_box([2,3,12,13],(40,40))==[0,0,24,25]
    assert context_box([20,20,80,40],(100,80))==[0,8,100,52]


def test_hybrid_is_gated_and_does_not_change_high_confidence_prediction():
    assert hybrid_choice('mosfet_pnp',.5,'bjt',.99)=='mosfet_pnp'
    assert hybrid_choice('mosfet_pnp',.49,'bjt',.79)=='mosfet_pnp'
    assert hybrid_choice('mosfet_pnp',.49,'bjt',.8)=='bjt'


def test_raw_subtype_survives_semantic_association_without_prefix_override():
    rows=[{'bbox':[30,30,60,60],'type':'mosfet_pnp','score':.3}]
    texts=[{'text':'Q1','bbox':[30,17,45,26],'score':.99}]
    pred,trace=export_shared(rows,texts,100)
    assert next(iter(pred['components'].values()))['type']=='mosfet_pnp'
    assert trace[0]['raw_pred_type']=='mosfet_pnp'
    assert rows[0]['type']=='mosfet_pnp' and 'id' not in rows[0]


def test_unknown_dev_class_and_missed_objects_stay_in_denominators():
    pred=[{'bbox':[0,0,20,20],'type':'crystal_2pin'}]
    gt=[{'bbox':[0,0,20,20],'type':'crystal'}, {'bbox':[80,80,90,90],'type':'r'}]
    assert pair_counts(pred,gt,'iou50',typed=False)=={'tp':1,'fp':0,'fn':1}
    assert pair_counts(pred,gt,'iou50',typed=True)=={'tp':0,'fp':1,'fn':2}


def test_matching_maximizes_cardinality_instead_of_greedy_neighbor_choice():
    pred=[{'bbox':[0,0,10,10],'type':'r'}, {'bbox':[-15,0,-5,10],'type':'r'}]
    gt=[{'bbox':[0,0,10,10],'type':'r'}, {'bbox':[18,0,28,10],'type':'r'}]
    assert pair_counts(pred,gt,'maxcoord20',typed=True)=={'tp':2,'fp':0,'fn':0}
