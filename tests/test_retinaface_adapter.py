"""Focused tests for the AMD RetinaFace MobileNet 0.25 adapter."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from scripts.evaluation import compare_widerface_detectors as comparison


def _empty_outputs():
    loc = np.zeros((1, comparison.RETINAFACE_ANCHOR_COUNT, 4), dtype=np.float32)
    confidence = np.zeros((1, comparison.RETINAFACE_ANCHOR_COUNT, 2), dtype=np.float32)
    confidence[:, :, 0] = 20.0
    confidence[:, :, 1] = 0.0
    landmarks = np.zeros((1, comparison.RETINAFACE_ANCHOR_COUNT, 10), dtype=np.float32)
    return [loc, confidence, landmarks]


def _fake_session(outputs=None, providers=None, input_shape=None, output_shapes=None):
    outputs = _empty_outputs() if outputs is None else outputs
    providers = ["CPUExecutionProvider"] if providers is None else providers
    input_shape = ["batch", 608, 640, 3] if input_shape is None else input_shape
    output_shapes = (
        [["batch", 15960, 4], ["batch", 15960, 2], ["batch", 15960, 10]]
        if output_shapes is None else output_shapes
    )
    session = Mock()
    session.get_providers.return_value = providers
    session.get_inputs.return_value = [SimpleNamespace(
        name="RetinaFace::input_0", shape=input_shape, type="tensor(float)"
    )]
    session.get_outputs.return_value = [
        SimpleNamespace(name=f"out_{index}", shape=shape, type="tensor(float)")
        for index, shape in enumerate(output_shapes)
    ]
    session.run.return_value = outputs
    return session


class RetinaFacePreprocessingTests(unittest.TestCase):
    def test_preprocessing_uses_bgr_mean_padding_and_nhwc(self):
        image = np.full((100, 200, 3), (150, 160, 170), dtype=np.uint8)

        tensor, resize_ratio = comparison.retinaface_preprocess(image)

        self.assertEqual(tensor.shape, (1, 608, 640, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(resize_ratio, 3.2)
        np.testing.assert_allclose(tensor[0, 0, 0], [46.0, 43.0, 47.0])
        np.testing.assert_allclose(tensor[0, 400, 0], [-104.0, -117.0, -123.0])

    def test_preprocessing_preserves_portrait_aspect_and_pads_right(self):
        image = np.full((1209, 1024, 3), 200, dtype=np.uint8)

        tensor, resize_ratio = comparison.retinaface_preprocess(image)

        self.assertEqual(tensor.shape, (1, 608, 640, 3))
        self.assertAlmostEqual(resize_ratio, 608 / 1209)
        self.assertTrue(np.all(tensor[0, :, -1, :] == np.asarray([-104, -117, -123])))

    def test_rejects_empty_or_non_bgr_inputs(self):
        with self.assertRaisesRegex(ValueError, "non-empty three-channel BGR"):
            comparison.retinaface_preprocess(np.zeros((0, 0, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "non-empty three-channel BGR"):
            comparison.retinaface_preprocess(np.zeros((4, 4), dtype=np.uint8))


class RetinaFaceDecodingTests(unittest.TestCase):
    def test_decodes_box_landmarks_and_maps_back_by_resize_ratio(self):
        outputs = _empty_outputs()
        row, column = 38, 40
        anchor_index = (row * 80 + column) * 2
        outputs[1][0, anchor_index] = [0.0, 20.0]
        priors = comparison.retinaface_prior_boxes()
        prior = priors[anchor_index]

        predictions = comparison.decode_retinaface_outputs(
            outputs, resize_ratio=0.5, original_width=1280, original_height=1216
        )

        self.assertEqual(len(predictions), 1)
        expected_box = np.asarray([
            (prior[0] - prior[2] / 2) * 640 / 0.5,
            (prior[1] - prior[3] / 2) * 608 / 0.5,
            (prior[0] + prior[2] / 2) * 640 / 0.5,
            (prior[1] + prior[3] / 2) * 608 / 0.5,
        ])
        np.testing.assert_allclose(predictions[0]["bbox"], expected_box, rtol=1e-6)
        self.assertAlmostEqual(predictions[0]["confidence"], 1.0, places=6)
        self.assertEqual(len(predictions[0]["landmarks"]), 10)
        expected_point = [prior[0] * 640 / 0.5, prior[1] * 608 / 0.5]
        np.testing.assert_allclose(
            np.asarray(predictions[0]["landmarks"]).reshape(5, 2),
            np.tile(expected_point, (5, 1)), rtol=1e-6,
        )

    def test_empty_confidence_set_returns_no_detections(self):
        predictions = comparison.decode_retinaface_outputs(
            _empty_outputs(), 1.0, original_width=640, original_height=608
        )
        self.assertEqual(predictions, [])

    def test_rejects_wrong_output_shapes_and_nonfinite_values(self):
        outputs = _empty_outputs()
        outputs[2] = outputs[2][:, :, :8]
        with self.assertRaisesRegex(RuntimeError, "Malformed RetinaFace outputs"):
            comparison.decode_retinaface_outputs(outputs, 1.0, 640, 608)

        outputs = _empty_outputs()
        outputs[0][0, 0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite output"):
            comparison.decode_retinaface_outputs(outputs, 1.0, 640, 608)

    def test_rejects_invalid_resize_ratio(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            comparison.decode_retinaface_outputs(_empty_outputs(), 0, 640, 608)


class RetinaFaceAdapterTests(unittest.TestCase):
    def test_adapter_uses_cpu_session_and_expected_input(self):
        session = _fake_session()
        adapter = comparison.RetinaFaceAdapter(session)

        predictions = adapter.detect(np.zeros((100, 200, 3), dtype=np.uint8))

        self.assertEqual(predictions, [])
        feed = session.run.call_args.args[1]
        self.assertIn("RetinaFace::input_0", feed)
        self.assertEqual(feed["RetinaFace::input_0"].shape, (1, 608, 640, 3))
        self.assertEqual(session.get_providers.return_value, ["CPUExecutionProvider"])

    def test_rejects_non_cpu_session_and_malformed_model_metadata(self):
        with self.assertRaisesRegex(ValueError, "CPUExecutionProvider only"):
            comparison.RetinaFaceAdapter(_fake_session(providers=["CUDAExecutionProvider"]))
        with self.assertRaisesRegex(RuntimeError, r"expected \[N,15960,10\]"):
            comparison.RetinaFaceAdapter(_fake_session(output_shapes=[
                ["batch", 15960, 4], ["batch", 15960, 2], ["batch", 15960, 8]
            ]))


if __name__ == "__main__":
    unittest.main()
