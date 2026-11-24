# FaceFind model rights and provenance

A code license, checkpoint distribution terms, training-data permissions and evaluation-image rights are separate. This register records the observed upstream notices and local file fingerprints. It does not grant new rights. Commercial deployment clearance has not been established for FaceFind.

## Bundled checkpoints

| Model and architecture | Exact checkpoint | Size in bytes | SHA-256 | Source and license |
|---|---|---:|---|---|
| YuNet lightweight detector, five landmarks | `face_detection_yunet_2023mar.onnx` | 232,589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | [OpenCV Zoo model distribution](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), [bundled MIT notice](LICENSE-YuNet.txt). |
| SFace MobileFaceNet embedding model, 128 dimensions | `face_recognition_sface_2021dec.onnx` | 38,696,353 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | [OpenCV Zoo model distribution](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), [bundled Apache-2.0 notice](LICENSE-SFace.txt). |

The repository carries the two checkpoint files with the model-directory notices. Those notices govern the upstream distributions. The weight creation and underlying training data, particularly SFace training provenance and any downstream restrictions, were not independently audited. A product deployment still requires review.

## Optional and experimental checkpoints

| Model | Checkpoint and local bytes | SHA-256 | Source, code and checkpoint scope |
|---|---|---|---|
| SCRFD-10G-KPS, five landmarks | `scrfd_10g_bnkps.onnx`, 16,923,827 | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` | [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo), also packaged in antelopev2. InsightFace code is MIT. Its published pretrained-weight terms restrict use to non-commercial research unless separately authorized. [Upstream terms](https://github.com/deepinsight/insightface/blob/master/server/LICENSING.md). |
| AdaFace IR-18, 512 dimensions | Local `adaface_ir_18_webface4m_yakhyo.onnx`, release `adaface_ir_18.onnx`, 96,105,345 | `6b6a35772fb636cdd4fa86520c1a259d0c41472a76f70f802b351837a00d9870` | [AdaFace research code](https://github.com/mk-minchul/AdaFace) and [third-party ONNX release](https://github.com/yakhyo/adaface-onnx/releases/tag/weights). The export repository has MIT code terms. A separate checkpoint grant and WebFace4M training-data rights were not established. |
| ArcFace ResNet100, 512 dimensions | `glintr100.onnx`, 260,665,334 | `4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf` | [InsightFace antelopev2](https://github.com/deepinsight/insightface/tree/master/model_zoo), Glint360K provenance as recorded. MIT code and non-commercial research pretrained-weight terms. |
| RetinaFace MobileNet 0.25 quantized | Upstream `RetinaFace_int.onnx`, local `retinaface_int.onnx`, 1,767,506 | `3b29de703b4641cb995754bc6294bd58b0fa5d859c838e9484c86d00251f8a08` | [AMD model card](https://huggingface.co/amd/retinaface) and [checkpoint](https://huggingface.co/amd/retinaface/resolve/main/weights/RetinaFace_int.onnx). Card declares Apache-2.0. Separate upstream training/weight rights review was not performed. |
| YOLOv8n-Face | `yolov8n-face.onnx`, 12,364,932 | `33f3951af7fc0c4d9b321b29cdcd8c9a59d0a29a8d4bdc01fcb5507d5c714809` | [Upstream inference repository](https://github.com/yakhyo/yolov8-face-onnx-inference) and [weight asset](https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/weights/yolov8n-face.onnx). Saved ONNX metadata declares AGPL-3.0. A separate license for the export repository code was not established. It reports WIDER FACE training. |

The optional and experimental models above are excluded from distribution. SHA-256 values fingerprint local files and are not necessarily an upstream published checksum. The selected SCRFD + AdaFace configuration is a local research/demo configuration. Obtain and use any weights only under appropriate authorization. Do not infer permission to redistribute AdaFace weights from the repository's code license.

The Milestone 5 runnable evaluator snapshot used only the two bundled checkpoints. The separate source review archive excludes model binaries. Locally obtained SCRFD and AdaFace checkpoints were used in the isolated preferred-pipeline test. The test does not establish a redistribution or commercial-use grant. Users supply optional checkpoints separately, subject to the applicable terms.

## Dataset materials

[WIDER FACE](https://shuoyang1213.me/WIDERFACE/) and [LFW](https://vis-www.cs.umass.edu/lfw/) were evaluation inputs under their respective terms. The raw image sets, derived LFW composites, private references and group photos, identity manifests and embeddings are excluded from the intended submission. The [research record](../../RESEARCH.md) contains aggregate results and describes usage. No training-data rights, redistribution rights for image sets, or clearance for biometric deployment are represented by this register.
