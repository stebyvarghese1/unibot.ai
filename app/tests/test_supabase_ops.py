import unittest
import os
import random
import string
from app.services.supabase_service import SupabaseService


class SupabaseOpsTestCase(unittest.TestCase):
    def setUp(self):
        # Ensure credentials exist
        self.supa = SupabaseService()
        self.local_pdf = os.path.join(os.getcwd(), "Product-Requirements-Document.pdf")
        if not os.path.exists(self.local_pdf):
            self.skipTest("Local test PDF not found")
        # Random suffix to avoid collisions
        self.suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        self.test_pdf_path = f"testing/prd-{self.suffix}.pdf"
        self.test_json_path = f"testing/chunks-{self.suffix}.json"

    def tearDown(self):
        # Cleanup in case a test failed mid-way
        try:
            self.supa.delete_file(self.test_pdf_path)
        except Exception:
            pass
        try:
            self.supa.delete_file(self.test_json_path)
        except Exception:
            pass

    def test_storage_upload_download_delete(self):
        # Upload PDF
        with open(self.local_pdf, "rb") as f:
            data = f.read()
        uploaded_path = self.supa.upload_file(data, self.test_pdf_path, content_type="application/pdf")
        self.assertEqual(uploaded_path, self.test_pdf_path)

        # Download and verify non-empty
        dl = self.supa.download_file(self.test_pdf_path)
        self.assertTrue(len(dl) > 0)

        # Upload JSON (simulating chunks audit file)
        payload = b'{"chunks": [{"chunk_index": 0, "text": "dummy"}]}'
        uploaded_json = self.supa.upload_file(payload, self.test_json_path, content_type="application/json")
        self.assertEqual(uploaded_json, self.test_json_path)

        # List and confirm presence
        items = self.supa.list_files(prefix="testing")
        names = set([(it.get("name") or it.get("Key") or "") for it in items])
        self.assertIn(self.test_pdf_path, names)
        self.assertIn(self.test_json_path, names)

        # Delete both
        self.assertTrue(self.supa.delete_file(self.test_pdf_path))
        self.assertTrue(self.supa.delete_file(self.test_json_path))

        # List again; ensure removed (eventual consistency should be immediate)
        items2 = self.supa.list_files(prefix="testing")
        names2 = set([(it.get("name") or it.get("Key") or "") for it in items2])
        self.assertNotIn(self.test_pdf_path, names2)
        self.assertNotIn(self.test_json_path, names2)


if __name__ == '__main__':
    unittest.main()
