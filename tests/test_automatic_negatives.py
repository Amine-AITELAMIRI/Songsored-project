import unittest
from songsored.negative_candidates import clean_anchor, finite_span, pick_pools


class AutomaticNegativeTests(unittest.TestCase):
    def setUp(self):
        self.words = ['one', 'two', 'safe', 'four', 'five']
        self.align = {str(i): dict(start=i*.3, end=i*.3+.2) for i in range(5)}

    def test_clean_context_and_guarded_target(self):
        self.assertTrue(clean_anchor(self.words, self.align, 2, set()))
        self.assertFalse(clean_anchor(self.words, self.align, 2, {4}))
        self.assertFalse(clean_anchor(self.words, self.align, 0, set()))

    def test_missing_or_disordered_anchors_are_rejected(self):
        self.align.pop('3')
        self.assertFalse(clean_anchor(self.words, self.align, 2, set()))
        self.align['3'] = dict(start=.1, end=.2)
        self.assertFalse(clean_anchor(self.words, self.align, 2, set()))

    def test_unknown_lyrics_and_nonfinite_bounds_are_rejected(self):
        self.words[1] = '_gap_'
        self.assertFalse(clean_anchor(self.words, self.align, 2, set()))
        self.assertFalse(finite_span(dict(start=True, end=2)))
        self.assertFalse(finite_span(dict(start=0, end=float('nan'))))
        self.assertFalse(finite_span(dict(start=2, end=1)))

    def test_selection_is_recording_diverse_and_spaced(self):
        pools = {rid: [dict(start=t, anchor_token=i,
                     category='similar_sound' if i%2 else 'ordinary_vocals')
                     for i,t in enumerate([1,1.2,7,13,19])]
                 for rid in ['a','b','c','d']}
        first = pick_pools(pools, 9, dict(similar_sound=4, ordinary_vocals=5))
        self.assertEqual(first, pick_pools(pools, 9, dict(similar_sound=4, ordinary_vocals=5)))
        self.assertEqual(sum(map(len, first.values())), 9)
        self.assertEqual(len(first), 4)
        for clips in first.values():
            self.assertLessEqual(len(clips), 3)
            self.assertTrue(all(abs(a['start']-b['start'])>=5
                                for i,a in enumerate(clips) for b in clips[i+1:]))


if __name__ == '__main__':
    unittest.main()
