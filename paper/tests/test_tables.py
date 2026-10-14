import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "paper_tables", Path(__file__).resolve().parents[1] / "build_tables.py"
)
tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tables)


class PaperTableTests(unittest.TestCase):
    def test_real_receipts_generate_all_run_cells(self):
        data, _ = tables.load_evidence()
        generated = tables.build(data)
        self.assertEqual(
            set(generated),
            {
                "design.tex",
                "adaptation.tex",
                "timing.tex",
                "per_cell.tex",
                "conventions.tex",
                "licences.tex",
                "receipts.tex",
            },
        )
        self.assertEqual(
            generated["per_cell.tex"].count("/metrics/dev/by_factor_cell_parity/"),
            11 * 40 * 5,
        )
        self.assertIn("Unresolved", generated["conventions.tex"])
        self.assertIn("credited under CC BY 4.0", generated["licences.tex"])
        self.assertNotIn("CC BY 4.0 terms", generated["licences.tex"])
        self.assertNotIn("version unresolved", generated["licences.tex"])
        self.assertIn(
            "PHUONG_RELEASE_STATUS_2026-10-12.complete.json", generated["licences.tex"]
        )
        self.assertEqual(generated, tables.build(data))

    def test_phuong_status_requires_the_bound_author_decision(self):
        data, _ = tables.load_evidence()
        status = data[tables.PHUONG_STATUS]
        status["decision_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "author decision"):
            tables.build(data)

    def test_current_attribution_must_match_author_status(self):
        data, _ = tables.load_evidence()
        status = data[tables.PHUONG_STATUS]
        data[status["attribution_path"]] = "stale attribution"
        with self.assertRaisesRegex(ValueError, "author decision"):
            tables.build(data)

    def test_historical_attribution_stays_bound_to_its_archived_bytes(self):
        data, index = tables.load_evidence()
        name = "docs/records/BENCHMARK_ATTRIBUTIONS_2026-09-10.md"
        prior = index["files"][name]["prior_version"]
        original = data["docs/records/RELEASE_LICENCE_REVIEW.complete.json"][
            "artifacts_sha256"
        ]["BENCHMARK_ATTRIBUTIONS_2026-09-10.md"]
        self.assertEqual(index["files"][prior]["sha256"], original)
        self.assertFalse(index["files"][prior]["derived_export_view"])
        self.assertIn("do not release this family", data[prior])
        self.assertNotIn("do not release this family", data[name])
        self.assertIn("render-attribution gate", data[name])

    def test_rebound_archive_cannot_satisfy_historical_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text("{}")
            (root / "prior.json").write_text('{"changed": true}')
            (root / "data.complete.json").write_text(
                json.dumps(
                    {
                        "artifacts_sha256": {
                            "data.json": tables.digest(root / "prior.json")
                        }
                    }
                )
            )
            records = {
                p.name: {"sha256": tables.digest(p), "derived_export_view": False}
                for p in root.glob("*.json")
            }
            records["data.json"]["prior_version"] = "prior.json"
            (root / "index.json").write_text(json.dumps({"files": records}))
            tables.load_evidence(root)
            records["prior.json"]["derived_export_view"] = True
            (root / "index.json").write_text(json.dumps({"files": records}))
            with self.assertRaisesRegex(ValueError, "Historical receipt"):
                tables.load_evidence(root)

    def test_decimal_display_rounding_matches_reported_precision(self):
        self.assertEqual(tables.percent_text(76.625), "76.63")
        self.assertEqual(tables.percent_text(15.475), "15.48")
        self.assertEqual(tables.percent_text(79.025), "79.03")
        self.assertEqual(tables.percent_text(99.12), "99.12")

    def test_every_cell_has_source_comment(self):
        value = tables.row([8000, 99.12, "a1_dark"], "results.complete.json", "metrics")
        self.assertEqual(
            value.count(
                "% receipt: evidence/results.complete.json; source: evidence/results.complete.json; metrics"
            ),
            3,
        )
        self.assertIn(r"a1\_\allowbreak{}dark", value)

    def test_tampered_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example.json").write_text("{}")
            (root / "index.json").write_text(
                json.dumps({"files": {"example.json": {"sha256": "0" * 64}}})
            )
            with self.assertRaisesRegex(ValueError, "Evidence hash"):
                tables.load_evidence(root)

    def test_historical_hash_link_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text("{}")
            (root / "data.complete.json").write_text(
                json.dumps({"artifacts_sha256": {"data.json": "1" * 64}})
            )
            (root / "index.json").write_text(
                json.dumps(
                    {
                        "files": {
                            p.name: {
                                "sha256": tables.digest(p),
                                "derived_export_view": False,
                            }
                            for p in root.glob("*.json")
                        }
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "Historical receipt"):
                tables.load_evidence(root)

    def test_withheld_digest_binds_only_a_copy_the_export_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text("{}")

            def load(bound, changed):
                (root / "data.complete.json").write_text(
                    json.dumps({"artifacts_sha256": {"data.json": bound}})
                )
                records = {
                    p.name: {"sha256": tables.digest(p), "derived_export_view": False}
                    for p in (root / "data.json", root / "data.complete.json")
                }
                records["data.json"]["derived_export_view"] = changed
                (root / "index.json").write_text(json.dumps({"files": records}))
                return tables.load_evidence(root)

            load(tables.WITHHELD, True)
            load(tables.digest(root / "data.json"), False)
            for bound, changed in (
                (tables.WITHHELD, False),
                (tables.digest(root / "data.json"), True),
            ):
                with self.assertRaisesRegex(ValueError, "Historical receipt"):
                    load(bound, changed)


if __name__ == "__main__":
    unittest.main()
