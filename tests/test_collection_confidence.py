"""Routing checks, not evidence of alignment accuracy."""
import copy
import math
import unittest
from songsored.candidate_routing import assess, route_frozen_selection


def example():
    candidate = dict(status='ok', stable=True, start=10., end=10.3,
                     character_support=.75, endpoint_spread=.02)
    return dict(selection={'status': 'ok', 'start': 10., 'end': 10.3},
                candidates={e: copy.deepcopy(candidate) for e in ('mms', 'native')})


class ConfidenceRoutingTests(unittest.TestCase):
    def test_provisional_label_never_claims_human_approval(self):
        result = route_frozen_selection(example(), 20)
        self.assertEqual(result['route'], 'auto_candidate')
        self.assertFalse(result['human_reviewed'])
        self.assertEqual(result['label_source'], 'automatic_unreviewed')

    def test_standard_frozen_acceptance_is_not_replaced_by_strict_diagnostic(self):
        case = example()
        case['candidates']['native']['character_support'] = .3
        result = route_frozen_selection(case, 20)
        self.assertEqual(result['route'], 'auto_candidate')
        self.assertEqual(result['confidence_band'], 'standard')

    def test_bad_selected_crop_never_enters_automatic_pool(self):
        for start, end, duration in [(None, 10.3, 20), (-1, .3, 20),
                (10, 11.1, 20), (10, 10.3, 10.1), (10, 10.3, float('nan'))]:
            with self.subTest(start=start, end=end, duration=duration):
                case = example()
                case['selection'].update(start=start, end=end)
                self.assertEqual(route_frozen_selection(case, duration)['route'], 'review_required')

    def test_agreement_is_uncalibrated_not_reviewed_truth(self):
        self.assertEqual(assess(example()), dict(route='auto_candidate', reasons=[], confidence_calibrated=False))

    def test_invalid_evidence_fails_closed(self):
        for field in ('start', 'end', 'character_support', 'endpoint_spread'):
            for value in (None, '0.9', True, math.nan, math.inf):
                with self.subTest(field=field, value=value):
                    case = example()
                    case['candidates']['mms'][field] = value
                    self.assertEqual(assess(case)['route'], 'review_required')

    def test_unreliable_evidence_requires_review(self):
        for field, value in [('stable', False), ('start', -1),
                ('character_support', .49), ('character_support', 1.1),
                ('endpoint_spread', -.1), ('endpoint_spread', .07),
                ('end', 10.01), ('end', 11)]:
            with self.subTest(field=field, value=value):
                case = example()
                case['candidates']['native'][field] = value
                self.assertEqual(assess(case)['route'], 'review_required')

    def test_missing_engine_and_abstention_require_review(self):
        case = example()
        del case['candidates']['native']
        self.assertEqual(assess(case)['route'], 'review_required')
        case = example()
        case['selection']['status'] = 'abstained'
        self.assertEqual(assess(case)['route'], 'review_required')

    def test_disagreement_requires_review_despite_high_scores(self):
        case = example()
        case['candidates']['native'].update(start=10.1, end=10.4)
        self.assertIn('model_disagreement', assess(case)['reasons'])

if __name__ == '__main__':
    unittest.main()
