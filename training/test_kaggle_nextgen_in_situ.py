import pathlib
import unittest


class KaggleNextGenInSituContractTest(unittest.TestCase):
    def test_in_situ_runner_keeps_authorization_and_build_before_gpu_runner(self):
        root = pathlib.Path(__file__).resolve().parent
        script = (root / "run_kaggle_nextgen_in_situ.sh").read_text(encoding="utf-8")
        self.assertIn('AETHEL_RUN_AUTHORIZED" != "YES"', script)
        self.assertIn('AETHEL_BUILD_DATA_IN_KAGGLE" != "YES"', script)
        self.assertIn("build_aethel_nextgen_data.sh", script)
        self.assertLess(script.index("build_aethel_nextgen_data.sh"), script.index("run_kaggle_nextgen.sh"))
        self.assertIn("AETHEL_PERSISTENCE_MODE:=notebook-output", script)

    def test_base_runner_accepts_versioned_notebook_output_without_dataset_token(self):
        root = pathlib.Path(__file__).resolve().parent
        script = (root / "run_kaggle_aethel.sh").read_text(encoding="utf-8")
        self.assertIn('AETHEL_PERSISTENCE_MODE:=kaggle-dataset', script)
        self.assertIn('"$AETHEL_PERSISTENCE_MODE" != "notebook-output"', script)
        self.assertIn("persistence_receipt.txt", script)


if __name__ == "__main__":
    unittest.main()
