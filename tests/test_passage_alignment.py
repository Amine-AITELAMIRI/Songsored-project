import unittest
import numpy as np
from songsored.ctc_alignment import passage,align_local,choose


class PassageTests(unittest.TestCase):
    def test_passage_does_not_use_target_coordinates(self):
        rows=[{'original':s} for s in ['left','target','right']]
        words={0:{'start':10.,'end':10.5},1:{'start':100.,'end':101.},2:{'start':11.5,'end':12.}}
        c={'token_start':1,'token_end':2}
        p=passage(rows,words,c,200)
        self.assertEqual((p['start'],p['end']),(10.,12.))
        words[1]={'start':1.,'end':2.}
        self.assertEqual(p,passage(rows,words,c,200))

    def test_bad_context_cannot_be_rescued_by_model_agreement(self):
        rows=[{'original':s} for s in ['stretched','left','target','right']]
        words={0:{'start':0.,'end':11.},1:{'start':11.,'end':11.5},3:{'start':12.,'end':12.5}}
        p=passage(rows,words,{'token_start':2,'token_end':3},20)
        candidate={'status':'ok','start':11.6,'end':11.9,'stable':True,'character_support':.9}
        self.assertEqual(choose({'mms':candidate,'native':candidate},p,20)['status'],'unresolved')

    def test_local_alignment_preserves_absolute_times_and_identity(self):
        import pytest
        pytest.importorskip("torchaudio")
        import torch
        # Repeated target far outside the anchored passage must not be selected.
        em=torch.full((1000,3),-12.);em[:,0]=0.
        em[210:216,1]=2.;em[230:236,2]=2.;em[800:810,1]=4.
        em=em.log_softmax(-1)
        times=np.column_stack([np.arange(1000)/20,np.arange(1,1001)/20])
        blob={'emission':em,'times':times,'dictionary':{'a':1,'b':2},'blank':0,'engine':'native','language':'en'}
        rows=[{'original':'a','native':'a','mms':'a','language':'en'},{'original':'b','native':'b','mms':'b','language':'en'}]
        p={'status':'ok','start':10.,'end':12.,'lo':0,'hi':2}
        result=align_local(blob,rows,{'token_start':0,'token_end':1},p,50.)
        self.assertEqual(result['status'],'ok');self.assertTrue(result['stable'])
        self.assertGreaterEqual(result['start'],10.);self.assertLess(result['end'],12.)

    def test_envelope_respects_neighbor_centers_and_stability(self):
        p={'status':'ok','warnings':[],'left_neighbor':{'start':1.,'end':1.3},'right_neighbor':{'start':1.5,'end':1.7}}
        m={'status':'ok','start':1.3,'end':1.5,'stable':True,'character_support':.7}
        n={**m,'start':1.25,'end':1.55}
        s=choose({'mms':m,'native':n},p,3.)
        self.assertAlmostEqual(s['start'],1.19);self.assertAlmostEqual(s['end'],1.6)
        n['stable']=False
        self.assertEqual(choose({'mms':m,'native':n},p,3.)['status'],'unresolved')


if __name__=='__main__':unittest.main()
