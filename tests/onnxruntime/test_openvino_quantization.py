#  Copyright 2026 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from optimum.openvino.quantization import OpenVINOQuantizer


class OpenVINOQuantizerDelegationTest(unittest.TestCase):
    def test_delegates_onnx_quantization_to_optimum_intel_ovquantizer(self):
        model_path = Path("tests/assets/onnx/model.onnx")

        with patch("optimum.openvino.quantization.INTEL_QUANTIZER_AVAILABLE", True):
            quantizer = OpenVINOQuantizer.from_pretrained(model_path)

        sample = {}
        for input_name in quantizer.input_names:
            shape = []
            for dim in quantizer.input_shapes[input_name]:
                shape.append(dim if isinstance(dim, int) and dim > 0 else 1)
            sample[input_name] = np.zeros(tuple(shape) if shape else (1,), dtype=np.int64)

        ov_config = SimpleNamespace(quantization_config=SimpleNamespace())
        calibration_data = [sample]

        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_intel_quantizer = Mock()
            with patch(
                "optimum.openvino.quantization.IntelOVQuantizer.from_pretrained",
                return_value=fake_intel_quantizer,
            ) as mock_from_pretrained:
                quantizer._quantize_with_ov(
                    ov_config=ov_config,
                    calibration_data=calibration_data,
                    save_directory=tmp_dir,
                )

            mock_from_pretrained.assert_called_once_with(
                model_path,
                seed=quantizer.seed,
                trust_remote_code=quantizer.trust_remote_code,
            )
            fake_intel_quantizer.quantize.assert_called_once_with(
                calibration_dataset=calibration_data,
                ov_config=ov_config,
                save_directory=tmp_dir,
                file_name=model_path.name,
            )
