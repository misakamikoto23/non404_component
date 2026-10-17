import unittest
from scripts.run_component_identity_experiment import validate_manifest


class IdentityManifestTests(unittest.TestCase):
    def test_duplicate_dev_job_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_manifest({"complete": True, "jobs": [{"id": "0001"}, {"id": "0001"}]}, {"0001"})

    def test_exact_dev_membership_is_required(self):
        self.assertEqual(len(validate_manifest({"complete": True, "jobs": [{"id": "0001"}]}, {"0001"})), 1)
        with self.assertRaises(ValueError):
            validate_manifest({"complete": True, "jobs": [{"id": "0002"}]}, {"0001"})
