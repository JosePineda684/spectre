# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import numpy.testing as npt
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.IO.H5.CombineH5Dat import combine_h5_dat
from spectre.Pipelines.Bbh.Cce import run_cce, run_cce_command


class TestCce(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/Cce"
        )
        print(self.test_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(unit_test_build_path(), "../../bin").resolve()

        # Set up directories to hold input and output files
        self.inspiral_dir = self.test_dir / "Inspiral"
        self.ringdown_dir = self.test_dir / "Ringdown"
        self.inspiral_seg_dir = self.inspiral_dir / "Segment_0000"
        self.ringdown_seg_dir = self.ringdown_dir / "Segment_0000"
        if os.path.exists(self.inspiral_dir):
            shutil.rmtree(self.inspiral_dir)
        if os.path.exists(self.ringdown_dir):
            shutil.rmtree(self.ringdown_dir)
        os.makedirs(self.inspiral_dir, exist_ok=True)
        os.makedirs(self.ringdown_dir, exist_ok=True)
        os.makedirs(self.inspiral_seg_dir, exist_ok=True)
        os.makedirs(self.ringdown_seg_dir, exist_ok=True)

        self.inspiral_input_file_path = os.path.join(
            self.inspiral_seg_dir, "BondiSachsCceR0200.h5"
        )
        self.ringdown_input_file_path = os.path.join(
            self.ringdown_seg_dir, "BondiSachsCceR0200.h5"
        )

        # Make BondiSachs data for CCE
        self.wave_inspiral_1 = np.array(
            [[t, np.sin(t), np.cos(t)] for t in np.arange(0, 10.0, 0.1)]
        )
        self.wave_inspiral_2 = np.array(
            [[t, 2 * np.sin(t), 2 * np.cos(t)] for t in np.arange(0, 10.0, 0.1)]
        )
        self.wave_ringdown_1 = np.array(
            [[t, np.sin(t), np.cos(t)] for t in np.arange(9, 14.0, 0.1)]
        )
        self.wave_ringdown_2 = np.array(
            [[t, 2 * np.sin(t), 2 * np.cos(t)] for t in np.arange(9, 14.0, 0.1)]
        )

        # Generate 2 h5 files, one for each segment, with two dat files each
        with spectre_h5.H5File(
            file_name=self.inspiral_input_file_path, mode="r+"
        ) as h5file:  # change lengend to be simialar to BondiSachs
            beta_datfile = h5file.insert_dat(
                path="/Beta", legend=["Time", "Re(0,0)", "Re(1,0)"], version=0
            )
            beta_datfile.append(self.wave_inspiral_1)
        with spectre_h5.H5File(
            file_name=self.inspiral_input_file_path, mode="r+"
        ) as h5file:
            drj_datfile = h5file.insert_dat(
                path="/DrJ",
                legend=["Time", "Re(0,0)", "Re(1,0)"],
                version=0,
            )
            drj_datfile.append(self.wave_inspiral_2)
        with spectre_h5.H5File(
            file_name=self.ringdown_input_file_path, mode="r+"
        ) as h5file:
            beta_datfile = h5file.insert_dat(
                path="/Beta", legend=["Time", "Re(0,0)", "Re(1,0)"], version=0
            )
            beta_datfile.append(self.wave_ringdown_1)
        with spectre_h5.H5File(
            file_name=self.ringdown_input_file_path, mode="r+"
        ) as h5file:
            drj_datfile = h5file.insert_dat(
                path="/DrJ",
                legend=["Time", "Re(0,0)", "Re(1,0)"],
                version=0,
            )
            drj_datfile.append(self.wave_ringdown_2)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cli(self):
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        try:
            run_cce_command(
                [
                    "--inspiral-run-dir",
                    str(self.inspiral_dir),
                    "--ringdown-run-dir",
                    str(self.ringdown_dir),
                    "--run-dir",
                    str(self.test_dir / "01_Output"),
                    "--executable",
                    str(self.bin_dir / "CharacteristicExtract"),
                    "--no-submit",
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        self.assertTrue((self.test_dir / "01_Output/Cce.yaml").exists())
        self.assertTrue(
            (self.test_dir / "01_Output/combinedBondiSachsCceR0200.h5").exists()
        )
        try:
            run_cce_command(
                [
                    "--bondisachs-data",
                    self.inspiral_input_file_path,
                    "--run-dir",
                    str(self.test_dir / "02_Output"),
                    "-E",
                    str(self.bin_dir / "CharacteristicExtract"),
                    "--no-submit",
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        self.assertTrue((self.test_dir / "02_Output/Cce.yaml").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
