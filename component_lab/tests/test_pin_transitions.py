import unittest
import cv2
import numpy as np
from component_lab.pin_transitions import infer_color_transitions,stable_color_change,TransitionConfig


class PinTransitionTests(unittest.TestCase):
    def make(self,side='right',width=1,colors=((0,0,180),(0,180,0))):
        image=np.full((140,140,3),255,dtype=np.uint8)
        box=[40,40,80,80]
        if side=='right':a,b,c=(81,60),(95,60),(125,60)
        elif side=='left':a,b,c=(39,60),(25,60),(5,60)
        elif side=='top':a,b,c=(60,39),(60,25),(60,5)
        else:a,b,c=(60,81),(60,95),(60,125)
        cv2.line(image,a,b,colors[0],width);cv2.line(image,b,c,colors[1],width)
        return image,[{'id':'owner','bbox':box}],b

    def test_four_directions_and_palette_independent(self):
        for side in ['left','right','top','bottom']:
            for colors in [((0,0,180),(0,180,0)),((40,40,40),(180,0,0)),((0,110,230),(180,0,180))]:
                im,boxes,b=self.make(side,colors=colors)
                candidates=infer_color_transitions(im,boxes)['candidates']
                self.assertEqual(len(candidates),1,(side,colors))
                self.assertLess(np.linalg.norm(np.array(candidates[0]['point_top_left'])-b),2)

    def test_line_widths(self):
        for width in [1,3,5]:
            im,boxes,b=self.make(width=width)
            tips=infer_color_transitions(im,boxes)['candidates']
            self.assertEqual(len(tips),1,width)
            self.assertLessEqual(np.linalg.norm(np.array(tips[0]['point_top_left'])-b),4)

    def test_monochrome_is_unresolved(self):
        im,boxes,_=self.make(colors=((0,0,0),(0,0,0)))
        r=infer_color_transitions(im,boxes)
        self.assertEqual(r['candidates'],[])
        self.assertGreater(r['reason_counts'].get('no_observable_color_boundary',0),0)

    def test_branch_before_transition_not_reported_as_tip(self):
        im,boxes,_=self.make();cv2.line(im,(86,47),(86,73),(0,0,180),1)
        r=infer_color_transitions(im,boxes)
        self.assertEqual(r['candidates'],[])

    def test_nearby_pins_not_globally_merged(self):
        im,boxes,_=self.make()
        cv2.line(im,(81,64),(95,64),(0,0,180),1);cv2.line(im,(95,64),(125,64),(0,180,0),1)
        self.assertEqual(len(infer_color_transitions(im,boxes)['candidates']),2)

    def test_later_remote_color_return_does_not_erase_first_boundary(self):
        colors=np.array([[50,128,128]]*5+[[50,170,170]]*5+[[50,128,128]]*5+[[50,170,170]]*5)
        index,reason=stable_color_change(colors,TransitionConfig())
        self.assertEqual(index,5);self.assertEqual(reason,'stable_chromatic_transition')

    def test_white_gap_at_color_boundary_is_unresolved(self):
        im,boxes,_=self.make();im[60,95]=255
        self.assertEqual(infer_color_transitions(im,boxes)['candidates'],[])

    def test_pale_fill_without_line_is_not_a_seed(self):
        im=np.full((140,140,3),255,dtype=np.uint8)
        cv2.rectangle(im,(35,35),(85,85),(170,255,255),-1)
        r=infer_color_transitions(im,[{'id':'owner','bbox':[40,40,80,80]}])
        self.assertEqual(r['candidates'],[])
        self.assertEqual(r['reason_counts']['no_boundary_seed'],1)
