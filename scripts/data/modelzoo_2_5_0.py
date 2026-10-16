# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The 147 DX Model Zoo additions that dx-modelzoo has not published yet.

This module is DATA. It is the single source for every generated artifact:
``config/model_registry.json`` entries, ``tests/data/processor_specs.json`` entries,
the ``provisional`` section of ``tests/data/modelzoo_cv_tree.json`` and the
``scripts/modelzoo_manifest.json`` rows. Change a family name here and re-run
``scripts/add_modelzoo_2_5_0_models.py`` -- never edit the generated files.

Why "provisional": dx-modelzoo assigns ``(task, family)`` as DATA, and for these 147
that data does not exist yet (harvested 2026-09-22: 350 variants, zero overlap with
this list). The names below follow dx-modelzoo's own rule, read from the repo: the
family boundary is the POSTPROCESSING CHAIN.

    yolov8-s_640x640         family yolov8     postprocessing: yolov8_decode, nms
    yolov8-s_640x640_decode  family yolov8     postprocessing: yolov8_decode, nms
    yolov8-n_640x640_ppu     family yolo-ppu   postprocessing: ppu_nms

A different *compiled output* of the same architecture with the same postprocessing
stays in the architecture family (``_decode``); a different *postprocessing mechanism*
gets its own family, even though ``yolov8-n_640x640_ppu`` shares weights with
``yolov8-n_640x640`` (``yolo-ppu`` holds v3/v4/v5/v7/v8/v9/v10/11/v12/x side by side).
Pre-optimized models carry a wholly different chain -- a top-k row table, no dense
decode, no standard NMS -- so they get mechanism families: ``yolo_preopt`` and
``yolo_preopt_pose``. A ``_pre-optimized`` filename alone does NOT put a model there:
only the ten stems MEASURED to emit the table (yolo26 det and pose) belong to them, and
every other ``_pre-optimized`` stem emits its plain sibling's tensor and stays in its
architecture family. There is no ``yolo_preopt_seg`` -- see the note where it would be.

dx_app renders dx-modelzoo's ``-`` as ``_`` in directory and registry ``family``
values, as it already does for ``yolo-ppu`` -> ``yolo_ppu`` and ``stdc-seg`` ->
``stdc_seg``.
"""
from __future__ import annotations

import re

try:
    from data.modelzoo_2_5_0_zoo_rows import ZOO_ROWS
except ImportError:  # run directly as scripts/data/modelzoo_2_5_0.py
    from modelzoo_2_5_0_zoo_rows import ZOO_ROWS

# --------------------------------------------------------------------- geometry

# The zoo resolution column uses two spellings, both verified against the 349
# published rows: HxWxC (341 rows, e.g. 224x224x3, 1024x2048x3) and WxCxH (8 rows
# whose MIDDLE token is the channel count, e.g. 1280x3x736 --
# retinaface_mobilenetv1_736x1280 pins that reading: registry w=1280 h=736).
#
# The .dxnn stem's own axis order is NOT consistent -- it inherits whatever the
# upstream repo called it (PaddleSeg's pp-liteseg 960x720 is WxH, MMSeg's
# stdc1-seg50 512x1024 is HxW) -- so the normalised column is the source, not the stem.
_GEOMETRY_EXCEPTIONS = {
    # NO exception for dark-hrnet-w32_256x192. There WAS one here, arguing that the
    # zoo had transposed the row: its own vitpose-s_256x192 row says 256x192x3 while
    # dark_hrnet_w32_256x192 says 192x256x3, and both are MMPose top-down person crops,
    # so the dark-hrnet row looked like the typo.
    #
    # The model says otherwise. With DX-RT 3.5.0 able to load it, its input tensor is
    # [1, 192, 256, 3] (NHWC) -- H=192, W=256, landscape -- and its heatmap is
    # (1, 17, 48, 64), which at stride 4 means exactly that. The zoo row was right and
    # the reasoning from convention was wrong. Left as a comment because "the table
    # looks transposed" is a tempting argument to make again.
    # Not an image. The .dxnn takes [1, 77, 512] CLIP token embeddings, and
    # CLIPTextPostprocessor documents its own axes as input_width=77 (token length),
    # input_height=512 (embedding width). Keep the class's convention.
    "clip-text_resnet50_77x512_openai": (77, 512),
}


def resolution(variant: str) -> tuple[int, int]:
    """``(width, height)`` for *variant*, from the zoo's normalised column."""
    if variant in _GEOMETRY_EXCEPTIONS:
        return _GEOMETRY_EXCEPTIONS[variant]
    parts = [int(p) for p in re.findall(r"\d+", ZOO_ROWS[variant][1])]
    if len(parts) == 3 and parts[1] == 3 and parts[0] != 3:
        return parts[0], parts[2]           # W x C x H
    if len(parts) == 3:
        return parts[1], parts[0]           # H x W x C
    if len(parts) == 2:
        return parts[1], parts[0]           # H x W
    raise ValueError(f"{variant}: cannot read resolution {ZOO_ROWS[variant][1]!r}")


# --------------------------------------------------------------------- spec shapes

WH = [{"token": "input_width"}, {"token": "input_height"}]
WHC = WH + [{"token": "config"}]


def _proc(cls: str, args: list, **kwargs) -> dict:
    return {"class": cls, "args": args, "kwargs": kwargs, "config_overrides": {}}


# IPoseFactory declares get_num_keypoints abstract. A generated pose factory without it
# raises TypeError at construction, which is how every pose variant failed the first
# full sweep -- the pre-existing pose families carry the same method through
# extra_methods, extracted from their originals.
COCO_17_KEYPOINTS = [
    'def get_num_keypoints(self) -> int:\n'
    '        """COCO 17-point body keypoints."""\n'
    '        return 17'
]


LETTERBOX = _proc("LetterboxPreprocessor", WH)
SIMPLE = _proc("SimpleResizePreprocessor", WH)

# open_clip's canonical normalisation. SimpleResizePreprocessor applies mean/std to the
# RAW 0-255 resized image, so the [0,1] constants are pre-multiplied here; the two forms
# are algebraically identical, (v - 255m) / 255s == (v/255 - m) / s. The existing clip
# family carries the same constants in custom_ops.py, where its measurement is recorded:
# against host FP32 over 21 images, x/255 scored p5 cosine 0.4724 / mean 0.7901 and these
# constants 0.8350 / 0.8935. Getting this wrong relocates image embeddings into a
# different space from host-encoded text and degrades retrieval without ever erroring.
CLIP_SIMPLE = _proc(
    "SimpleResizePreprocessor", WH,
    mean=[0.48145466 * 255.0, 0.4578275 * 255.0, 0.40821073 * 255.0],
    std=[0.26862954 * 255.0, 0.26130258 * 255.0, 0.27577711 * 255.0],
)

# Manifest 'category' follows the legacy task spelling the existing 352 rows use.
MANIFEST_CATEGORY = {
    "object_detection": "Object Detection",
    "instance_segmentation": "Instance Segmentation",
    "pose_estimation": "Pose Estimation",
    "classification": "Classification",
    "reid": "Classification",
    "semantic_segmentation": "Semantic Segmentation",
    "embedding": "Embedding",
    "super_resolution": "Super Resolution",
    "anomaly_detection": "Anomaly Detection",
}

# Upstream references. The zoo leaves the Source column empty for all 143 new rows.
REF_ULTRALYTICS = "https://github.com/ultralytics/ultralytics"
REF_YOLOV9 = "https://github.com/WongKinYiu/yolov9"
REF_YOLOV12 = "https://github.com/sunsmarterjie/yolov12"
REF_YOLOV13 = "https://github.com/iMoonLab/yolov13"
REF_YOLOV6 = "https://github.com/meituan/YOLOv6"
REF_PADDLEDET = "https://github.com/PaddlePaddle/PaddleDetection"
REF_PADDLECLAS = "https://github.com/PaddlePaddle/PaddleClas"
REF_PADDLESEG = "https://github.com/PaddlePaddle/PaddleSeg"
REF_DDRNET = "https://github.com/ydhongHIT/DDRNet"
REF_STDC = "https://github.com/MichaelFan01/STDC-Seg"
REF_EFFICIENTAD = "https://github.com/amazon-science/efficient-ad"
REF_PATCHCORE = "https://github.com/amazon-science/patchcore-inspection"
REF_CLIP = "https://github.com/openai/CLIP"
REF_METACLIP = "https://github.com/facebookresearch/MetaCLIP"
REF_SWIN = "https://github.com/microsoft/Swin-Transformer"
REF_VIT = "https://github.com/google-research/vision_transformer"
REF_TORCHVISION = "https://github.com/pytorch/vision"
REF_HRNET = "https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation"


# C++ has no ast module, so the C++ generator never re-derives a factory -- it carries
# the donor variant's own factory over verbatim and rewrites only the class name,
# include guard and getModelName(). A variant introduced AFTER the restructure has no
# such original, so each group names the EXISTING C++ family whose factory is
# behaviourally right for it (``cpp_donor``): yolov12/yolov13 detection take yolov8's,
# the PaddleClas classifiers take resnet's, PP-ShiTu's mainbody detector takes
# nanodet's (both PicoDet/NanoDet-Plus GFL heads). ``None`` means no existing C++
# postprocessor fits and the family's factory is hand-written -- yolo_preopt*, rtdetr,
# mask_rtdetr, ppmatting, efficientad, patchcore.

_SIZES = ("n", "s", "m", "l", "x")


def _preopt(arch: str, sizes, infix: str = "") -> list[str]:
    return [f"{arch}-{s}{infix}_640x640_pre-optimized" for s in sizes]


def _plain(arch: str, sizes, infix: str = "", res: str = "640x640") -> list[str]:
    return [f"{arch}-{s}{infix}_{res}" for s in sizes]


EFFICIENTAD_COMPANIONS = [
    'def get_companion_models(self, primary_path: str):\n'
    '        """The other two EfficientAD networks, beside the one ``-m`` named.\n'
    '\n'
    '        EfficientAD scores a DISAGREEMENT -- the student\'s first 384 channels\n'
    '        predict the teacher and its second 384 predict the autoencoder -- so no\n'
    '        single network can produce the map. Declaring the set on the factory keeps\n'
    '        the one-``-m`` CLI contract that run_demo.sh and every sweep rely on; the\n'
    '        runner resolves these names in the primary model\'s own directory.\n'
    '\n'
    '        The returned order matches ``config.roles`` in each variant config, which\n'
    '        is the only thing that tells two same-shaped 384-channel maps apart.\n'
    '        """\n'
    '        from pathlib import Path as _Path\n'
    '        roles = ("student", "teacher", "autoencoder")\n'
    '        stem = _Path(primary_path).name\n'
    '        primary = next((r for r in roles if f"-{r}_" in stem), None)\n'
    '        if primary is None:\n'
    '            raise ValueError(\n'
    '                f"{stem} does not name an EfficientAD role; expected one of "\n'
    '                f"{roles}.")\n'
    '        return [(other, stem.replace(f"-{primary}_", f"-{other}_"))\n'
    '                for other in roles if other != primary]'
]


# --------------------------------------------------------------------- the table
#
# Every group states its own task, family, processors and factory base. Nothing is
# inferred from the task at emit time: image_classification alone mixes
# IClassificationFactory and IEmbeddingFactory, so a per-task lookup would silently
# re-base a family.

GROUPS: list[dict] = [
    # ============================ pre-optimized (63) ============================
    dict(
        task="object_detection", task_legacy="object_detection", family="yolo_preopt",
        cpp_donor=None,
        # MEASURED on DX-RT 3.5.0, not assumed from the name. Of the 63 stems ending
        # in _pre-optimized, only these five emit the top-k row table (1, 300, 6). The
        # other 25 detection ones emit (1, 84, 8400) -- the ORDINARY dense head, the
        # same tensor their plain sibling emits -- so by dx-modelzoo's own rule
        # (family = postprocessing chain) they belong to their architecture family and
        # are listed there. "_pre-optimized" in the filename does not imply the
        # pre-optimized HEAD; for yolo11/v8/v9/v12/v13 it is a compile difference the
        # application never sees.
        variants=["yolo26-l_640x640_pre-optimized", "yolo26-m_640x640_pre-optimized", "yolo26-n_640x640_pre-optimized", "yolo26-s_640x640_pre-optimized", "yolo26-x_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("PreoptDetectionPostprocessor", WHC),
        visualizer=_proc("DetectionVisualizer", []),
        base="IDetectionFactory", reference=REF_ULTRALYTICS,
        config={"score_threshold": 0.3, "nms_threshold": 0.45, "top_k": 300},
    ),
    dict(
        task="pose_estimation", task_legacy="pose_estimation", family="yolo_preopt_pose",
        cpp_donor=None,
        # MEASURED: only these five emit (1, 300, 57). The other ten pose stems emit
        # (1, 56, 8400), the ordinary dense pose head.
        variants=["yolo26-l-pose_640x640_pre-optimized", "yolo26-m-pose_640x640_pre-optimized", "yolo26-n-pose_640x640_pre-optimized", "yolo26-s-pose_640x640_pre-optimized", "yolo26-x-pose_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("PreoptPosePostprocessor", WHC),
        visualizer=_proc("PoseVisualizer", []),
        base="IPoseFactory", reference=REF_ULTRALYTICS,
        config={"score_threshold": 0.3, "nms_threshold": 0.45, "top_k": 300,
                "num_keypoints": 17},
    ),
    # There is NO yolo_preopt_seg family. All 18 stems ending in -seg_..._pre-optimized
    # emit (1, 116, 8400) + (1, 32, 160, 160) -- the ordinary dense segmentation head --
    # so every one of them is a variant of its architecture's _seg family. The family
    # was created here on the assumption that the name implied the pre-optimized head;
    # it does not, and no published model emits a pre-optimized SEG table at all.
    # PreoptSegPostprocessor is kept (tested, and dx_yolo26 ships a model that needs
    # it), just not wired to any variant.

    # ============================ plain YOLO (25) ===============================
    dict(
        task="object_detection", task_legacy="object_detection", family="yolov12",
        cpp_donor="object_detection/yolov8",
        # The _pre-optimized stems of this family emit the SAME dense tensor as
        # the plain ones (measured), so they are variants of this family, not of a
        # separate mechanism family.
        variants=_plain("yolov12", _SIZES) + ["yolov12-l_640x640_pre-optimized", "yolov12-m_640x640_pre-optimized", "yolov12-n_640x640_pre-optimized", "yolov12-s_640x640_pre-optimized", "yolov12-x_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8Postprocessor", WHC),
        visualizer=_proc("DetectionVisualizer", []),
        base="IDetectionFactory", reference=REF_YOLOV12,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),
    dict(
        task="object_detection", task_legacy="object_detection", family="yolov13",
        cpp_donor="object_detection/yolov8",
        # The _pre-optimized stems of this family emit the SAME dense tensor as
        # the plain ones (measured), so they are variants of this family, not of a
        # separate mechanism family.
        variants=_plain("yolov13", ("n", "s", "l", "x")) + ["yolov13-l_640x640_pre-optimized", "yolov13-n_640x640_pre-optimized", "yolov13-s_640x640_pre-optimized", "yolov13-x_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8Postprocessor", WHC),
        visualizer=_proc("DetectionVisualizer", []),
        base="IDetectionFactory", reference=REF_YOLOV13,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),
    dict(
        task="object_detection", task_legacy="object_detection", family="yolov9",
        cpp_donor="object_detection/yolov9",
        # The _pre-optimized stems of this family emit the SAME dense tensor as
        # the plain ones (measured), so they are variants of this family, not of a
        # separate mechanism family.
        variants=["yolov9-e_640x640"] + ["yolov9-c_640x640_pre-optimized", "yolov9-e_640x640_pre-optimized", "yolov9-gelan-c_640x640_pre-optimized", "yolov9-m_640x640_pre-optimized", "yolov9-s_640x640_pre-optimized", "yolov9-t_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8Postprocessor", WHC),
        visualizer=_proc("DetectionVisualizer", []),
        base="IDetectionFactory", reference=REF_YOLOV9,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),
    dict(
        task="object_detection", task_legacy="object_detection", family="yolov6",
        cpp_donor="object_detection/yolov6",
        variants=["yolov6-m6_1280x1280"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8Postprocessor", WHC),
        visualizer=_proc("DetectionVisualizer", []),
        base="IDetectionFactory", reference=REF_YOLOV6,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),
    dict(
        task="instance_segmentation", task_legacy="instance_segmentation",
        family="yolov12_seg",
        cpp_donor="instance_segmentation/yolov8_seg",
        # The _pre-optimized stems of this family emit the SAME dense tensor as
        # the plain ones (measured), so they are variants of this family, not of a
        # separate mechanism family.
        variants=[f"yolov12-seg-{s}_640x640" for s in _SIZES] + ["yolov12-seg-l_640x640_pre-optimized", "yolov12-seg-m_640x640_pre-optimized", "yolov12-seg-n_640x640_pre-optimized", "yolov12-seg-s_640x640_pre-optimized", "yolov12-seg-x_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8InstanceSegPostprocessor", WHC),
        visualizer=_proc("InstanceSegVisualizer", []),
        base="IInstanceSegFactory", reference=REF_YOLOV12,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),
    dict(
        task="instance_segmentation", task_legacy="instance_segmentation",
        family="yolov9_seg",
        cpp_donor="instance_segmentation/yolov8_seg",
        # The _pre-optimized stems of this family emit the SAME dense tensor as
        # the plain ones (measured), so they are variants of this family, not of a
        # separate mechanism family.
        variants=[f"yolov9-seg-{s}_640x640" for s in ("c", "e", "gelan-c")] + ["yolov9-seg-c_640x640_pre-optimized", "yolov9-seg-e_640x640_pre-optimized", "yolov9-seg-gelan-c_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8InstanceSegPostprocessor", WHC),
        visualizer=_proc("InstanceSegVisualizer", []),
        base="IInstanceSegFactory", reference=REF_YOLOV9,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),
    dict(
        task="image_classification", task_legacy="classification", family="yolov12_cls",
        cpp_donor="image_classification/yolo26_cls",
        variants=[f"yolov12-cls-{s}_224x224" for s in _SIZES],
        preprocessor=SIMPLE,
        postprocessor=_proc("ClassificationPostprocessor", WHC),
        visualizer=_proc("ClassificationVisualizer", []),
        base="IClassificationFactory", reference=REF_YOLOV12,
        config={"top_k": 5},
    ),
    dict(
        # Gap fill: dx-modelzoo's yolo11-pose holds n/s/m/x but not l.
        task="pose_estimation", task_legacy="pose_estimation", family="yolo11_pose",
        cpp_donor="pose_estimation/yolo11_pose",
        # The _pre-optimized stems of this family emit the SAME dense tensor as
        # the plain ones (measured), so they are variants of this family, not of a
        # separate mechanism family.
        variants=["yolo11-l-pose_640x640"] + ["yolo11-l-pose_640x640_pre-optimized", "yolo11-m-pose_640x640_pre-optimized", "yolo11-n-pose_640x640_pre-optimized", "yolo11-s-pose_640x640_pre-optimized", "yolo11-x-pose_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8PosePostprocessor", WHC),
        visualizer=_proc("PoseVisualizer", []),
        base="IPoseFactory", reference=REF_ULTRALYTICS,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),

    dict(
        # An EXISTING dx-modelzoo family gaining its _pre-optimized stems: measured,
        # they emit the same dense tensor as the plain variants already here.
        task="object_detection", task_legacy="object_detection", family="yolo11",
        cpp_donor="object_detection/yolo11",
        variants=["yolo11-l_640x640_pre-optimized", "yolo11-m_640x640_pre-optimized", "yolo11-n_640x640_pre-optimized", "yolo11-s_640x640_pre-optimized", "yolo11-x_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8Postprocessor", WHC),
        visualizer=_proc("DetectionVisualizer", []),
        base="IDetectionFactory", reference=REF_ULTRALYTICS,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),
    dict(
        # An EXISTING dx-modelzoo family gaining its _pre-optimized stems: measured,
        # they emit the same dense tensor as the plain variants already here.
        task="object_detection", task_legacy="object_detection", family="yolov8",
        cpp_donor="object_detection/yolov8",
        variants=["yolov8-l_640x640_pre-optimized", "yolov8-m_640x640_pre-optimized", "yolov8-n_640x640_pre-optimized", "yolov8-s_640x640_pre-optimized", "yolov8-x_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8Postprocessor", WHC),
        visualizer=_proc("DetectionVisualizer", []),
        base="IDetectionFactory", reference=REF_ULTRALYTICS,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),
    dict(
        # An EXISTING dx-modelzoo family gaining its _pre-optimized stems: measured,
        # they emit the same dense tensor as the plain variants already here.
        task="instance_segmentation", task_legacy="instance_segmentation", family="yolo11_seg",
        cpp_donor="instance_segmentation/yolo11_seg",
        variants=["yolo11-l-seg_640x640_pre-optimized", "yolo11-m-seg_640x640_pre-optimized", "yolo11-n-seg_640x640_pre-optimized", "yolo11-s-seg_640x640_pre-optimized", "yolo11-x-seg_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8InstanceSegPostprocessor", WHC),
        visualizer=_proc("InstanceSegVisualizer", []),
        base="IInstanceSegFactory", reference=REF_ULTRALYTICS,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),
    dict(
        # An EXISTING dx-modelzoo family gaining its _pre-optimized stems: measured,
        # they emit the same dense tensor as the plain variants already here.
        task="instance_segmentation", task_legacy="instance_segmentation", family="yolov8_seg",
        cpp_donor="instance_segmentation/yolov8_seg",
        variants=["yolov8-l-seg_640x640_pre-optimized", "yolov8-m-seg_640x640_pre-optimized", "yolov8-n-seg_640x640_pre-optimized", "yolov8-s-seg_640x640_pre-optimized", "yolov8-x-seg_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8InstanceSegPostprocessor", WHC),
        visualizer=_proc("InstanceSegVisualizer", []),
        base="IInstanceSegFactory", reference=REF_ULTRALYTICS,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),
    dict(
        # An EXISTING dx-modelzoo family gaining its _pre-optimized stems: measured,
        # they emit the same dense tensor as the plain variants already here.
        task="pose_estimation", task_legacy="pose_estimation", family="yolov8_pose",
        cpp_donor="pose_estimation/yolov8_pose",
        variants=["yolov8-l-pose_640x640_pre-optimized", "yolov8-m-pose_640x640_pre-optimized", "yolov8-n-pose_640x640_pre-optimized", "yolov8-s-pose_640x640_pre-optimized", "yolov8-x-pose_640x640_pre-optimized"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("YOLOv8PosePostprocessor", WHC),
        visualizer=_proc("PoseVisualizer", []),
        base="IPoseFactory", reference=REF_ULTRALYTICS,
        config={"score_threshold": 0.4, "nms_threshold": 0.5},
    ),
    # ============================ RT-DETR (20) ==================================
    dict(
        task="object_detection", task_legacy="object_detection", family="rtdetr",
        cpp_donor=None,
        variants=["rtdetr-hgnetv2-h-6x_640x640", "rtdetr-hgnetv2-l-6x_640x640",
                  "rtdetr-hgnetv2-x-6x_640x640", "rtdetr-r101vd-6x_640x640",
                  "rtdetr-r18vd-6x_640x640", "rtdetr-r34vd-6x_640x640",
                  "rtdetr-r50vd-6x_640x640", "rtdetr-r50vd-m-6x_640x640",
                  "rtdetrv2-r101vd-6x_640x640", "rtdetrv2-r18vd-120e_640x640",
                  "rtdetrv2-r34vd-120e_640x640", "rtdetrv2-r50vd-6x_640x640",
                  "rtdetrv2-r50vd-m-7x_640x640", "rtdetrv3-r18vd-6x_640x640",
                  "rtdetrv3-r50vd-6x_640x640"],
        preprocessor=SIMPLE,
        postprocessor=_proc("RTDETRPostprocessor", WHC),
        visualizer=_proc("DetectionVisualizer", []),
        base="IDetectionFactory", reference=REF_PADDLEDET,
        # NMS-free by construction: nms_threshold 1.0 keeps every query. RT-DETR's own
        # preprocessing is a plain resize, so there is no letterbox to undo.
        config={"score_threshold": 0.4, "nms_threshold": 1.0},
    ),
    dict(
        task="instance_segmentation", task_legacy="instance_segmentation",
        family="mask_rtdetr",
        cpp_donor=None,
        variants=[f"mask-rtdetr-hgnetv2-{s}-6x_640x640"
                  for s in ("s", "m", "l", "x", "h")],
        preprocessor=SIMPLE,
        postprocessor=_proc("MaskRTDETRPostprocessor", WHC),
        visualizer=_proc("InstanceSegVisualizer", []),
        base="IInstanceSegFactory", reference=REF_PADDLEDET,
        config={"score_threshold": 0.4, "nms_threshold": 1.0},
    ),

    # ============================ classification (17) ===========================
    dict(
        task="image_classification", task_legacy="classification", family="pphgnet",
        cpp_donor="image_classification/resnet",
        variants=[f"pphgnet-{s}_224x224" for s in ("tiny", "small", "base")],
        preprocessor=SIMPLE,
        postprocessor=_proc("ClassificationPostprocessor", WHC),
        visualizer=_proc("ClassificationVisualizer", []),
        base="IClassificationFactory", reference=REF_PADDLECLAS,
        config={"top_k": 5},
    ),
    dict(
        task="image_classification", task_legacy="classification", family="pphgnetv2",
        cpp_donor="image_classification/resnet",
        variants=[f"pphgnetv2-b{i}_224x224" for i in range(7)],
        preprocessor=SIMPLE,
        postprocessor=_proc("ClassificationPostprocessor", WHC),
        visualizer=_proc("ClassificationVisualizer", []),
        base="IClassificationFactory", reference=REF_PADDLECLAS,
        config={"top_k": 5},
    ),
    dict(
        task="image_classification", task_legacy="classification", family="pplcnetv2",
        cpp_donor="image_classification/resnet",
        variants=[f"pplcnetv2-{s}_224x224" for s in ("small", "base", "large")],
        preprocessor=SIMPLE,
        postprocessor=_proc("ClassificationPostprocessor", WHC),
        visualizer=_proc("ClassificationVisualizer", []),
        base="IClassificationFactory", reference=REF_PADDLECLAS,
        config={"top_k": 5},
    ),
    dict(
        task="image_classification", task_legacy="classification", family="swin",
        cpp_donor="image_classification/vit",
        variants=["swin-b_224x224"],
        preprocessor=SIMPLE,
        postprocessor=_proc("ClassificationPostprocessor", WHC),
        visualizer=_proc("ClassificationVisualizer", []),
        base="IClassificationFactory", reference=REF_SWIN,
        config={"top_k": 5},
    ),
    dict(
        # Joins the existing dx-modelzoo 'vit' family (5 published variants).
        task="image_classification", task_legacy="classification", family="vit",
        cpp_donor="image_classification/vit",
        variants=["vit-l-p16_512x512_swag", "vit-b-p16_384x384", "vit-t-p16_224x224"],
        preprocessor=SIMPLE,
        postprocessor=_proc("ClassificationPostprocessor", WHC),
        visualizer=_proc("ClassificationVisualizer", []),
        base="IClassificationFactory", reference=REF_VIT,
        config={"top_k": 5},
    ),
    dict(
        task="image_classification", task_legacy="classification", family="beit",
        cpp_donor="image_classification/beit",
        variants=["beit-l-p16_384x384"],
        preprocessor=SIMPLE,
        postprocessor=_proc("ClassificationPostprocessor", WHC),
        visualizer=_proc("ClassificationVisualizer", []),
        base="IClassificationFactory", reference=REF_TORCHVISION,
        config={"top_k": 5},
    ),
    dict(
        task="image_classification", task_legacy="classification", family="levit",
        cpp_donor="image_classification/levit",
        variants=["levit-128s_224x224"],
        preprocessor=SIMPLE,
        postprocessor=_proc("ClassificationPostprocessor", WHC),
        visualizer=_proc("ClassificationVisualizer", []),
        base="IClassificationFactory", reference=REF_TORCHVISION,
        config={"top_k": 5},
    ),

    # ============================ segmentation (8) ==============================
    dict(
        task="semantic_segmentation", task_legacy="semantic_segmentation",
        family="ddrnet",
        cpp_donor="semantic_segmentation/bisenet",
        variants=["ddrnet23_1024x2048", "ddrnet23-slim_1024x2048"],
        preprocessor=SIMPLE,
        postprocessor=_proc("SemanticSegmentationPostprocessor", WHC),
        visualizer=_proc("SemanticSegmentationVisualizer", []),
        base="ISegmentationFactory", reference=REF_DDRNET,
        config={"num_classes": 19},
    ),
    dict(
        # Joins the existing dx-modelzoo 'stdc-seg' family (stdc2-seg50_512x1024).
        task="semantic_segmentation", task_legacy="semantic_segmentation",
        family="stdc_seg",
        cpp_donor="semantic_segmentation/stdc_seg",
        variants=["stdc1-seg50_512x1024", "stdc1-seg75_768x1536",
                  "stdc2-seg75_768x1536"],
        preprocessor=SIMPLE,
        postprocessor=_proc("SemanticSegmentationPostprocessor", WHC),
        visualizer=_proc("SemanticSegmentationVisualizer", []),
        base="ISegmentationFactory", reference=REF_STDC,
        config={"num_classes": 19},
    ),
    dict(
        task="semantic_segmentation", task_legacy="semantic_segmentation",
        family="pp_liteseg",
        cpp_donor="semantic_segmentation/bisenet",
        variants=["pp-liteseg-stdc1-camvid-10k_960x720"],
        preprocessor=SIMPLE,
        postprocessor=_proc("SemanticSegmentationPostprocessor", WHC),
        visualizer=_proc("SemanticSegmentationVisualizer", []),
        base="ISegmentationFactory", reference=REF_PADDLESEG,
        config={"num_classes": 11},          # CamVid
    ),
    dict(
        # The zoo lists these under Image Matting, not Semantic Segmentation, and the
        # distinction is not cosmetic: SemanticSegmentationVisualizer reads `mask` as
        # CLASS IDS and paints it with a 19-colour Cityscapes palette, which renders a
        # continuous alpha matte as a two-tone segmentation and discards the matte.
        task="image_matting", task_legacy="semantic_segmentation",
        family="ppmatting",
        cpp_donor=None,
        variants=["ppmatting-hrnet-w48-composition_512x512",
                  "ppmatting-hrnet-w48-distinctions_512x512"],
        # Gated illumination gain: a white shirt on a white studio wall (frame mean
        # 209) left the face and hands as EXACT zeros in the matte. See
        # SimpleResizePreprocessor.mean_target for the measurement; the gate keeps
        # every in-distribution subject bit-identical.
        preprocessor=_proc("SimpleResizePreprocessor", WH,
                           mean_target=110, mean_target_above=195),
        postprocessor=_proc("PPMattingPostprocessor", WHC),
        visualizer=_proc("MattingVisualizer", [{"token": "config"}]),
        base="ISegmentationFactory", reference=REF_PADDLESEG,
        config={"alpha_threshold": 0.5},
    ),

    # ============================ anomaly detection (4) =========================
    dict(
        task="anomaly_detection", task_legacy="anomaly_detection",
        family="efficientad",
        cpp_donor=None,
        variants=[f"efficientad-m-{p}_256x256"
                  for p in ("teacher", "student", "autoencoder")],
        preprocessor=SIMPLE,
        postprocessor=_proc("EfficientADPostprocessor", WHC),
        visualizer=_proc("AnomalyVisualizer", []),
        base="IAnomalyDetectionFactory", reference=REF_EFFICIENTAD,
        config={},
        # ``-m`` names the PRIMARY network and the companions follow it, so each
        # variant sees its outputs in a different order. Teacher and autoencoder are
        # both (1,384,H,W): only this declaration distinguishes them.
        config_by_variant={
            f"efficientad-m-{primary}_256x256": {
                "roles": [primary] + [r for r in ("student", "teacher", "autoencoder")
                                      if r != primary]}
            for primary in ("teacher", "student", "autoencoder")
        },
        extra_methods=EFFICIENTAD_COMPANIONS,
    ),
    dict(
        task="anomaly_detection", task_legacy="anomaly_detection", family="patchcore",
        cpp_donor=None,
        variants=["patchcore_224x224"],
        preprocessor=SIMPLE,
        postprocessor=_proc("AnomalyFeaturePostprocessor", WHC),
        visualizer=_proc("AnomalyVisualizer", []),
        base="IAnomalyDetectionFactory", reference=REF_PATCHCORE,
        config={},
    ),

    # ============================ CLIP / embedding (6) ==========================
    dict(
        # Zoo category: Image Retrieval holds this checkpoint's TEXT tower; its image
        # tower comes with it, because retrieval needs both halves of one embedding
        # space. MEASURED: this .dxnn takes uint8 [1,224,224,3], so CLIP_SIMPLE (which
        # emits float32) made the example unrunnable -- "Input dtype mismatch for 'x':
        # expected uint8, got float32". The normalisation is folded into the compiled
        # graph, as for every other uint8 model here.
        task="image_retrieval", task_legacy="embedding", family="clip_rn50",
        cpp_donor="image_retrieval/clip_rn50",
        variants=["clip-img_resnet50_224x224_openai"],
        preprocessor=SIMPLE,
        postprocessor=_proc("GalleryRetrievalPostprocessor", WHC,
                            model_type="image_embedding",
                            model_name="clip-img_resnet50_224x224_openai"),
        visualizer=_proc("RetrievalVisualizer", [{"token": "config"}]),
        base="IEmbeddingFactory", reference=REF_CLIP, config={},
    ),
    dict(
        task="zero_shot_image_classification", task_legacy="embedding", family="clip",
        cpp_donor="zero_shot_image_classification/clip",
        variants=["clip-img_vit-b16-quickgelu_224x224_metaclip-fullcc"],
        preprocessor=CLIP_SIMPLE,
        postprocessor=_proc("CLIPImagePostprocessor", WHC),
        visualizer=_proc("EmbeddingVisualizer", []),
        base="IEmbeddingFactory", reference=REF_METACLIP, config={},
    ),
    dict(
        # Text encoder: the .dxnn takes [1, 77, 512] token embeddings, not an image.
        # CONFIRMED against the model on DX-RT 3.5.0: its input dtype is float32, and
        # feeding it the uint8 tensor an image preprocessor produces fails outright
        # ("Input dtype mismatch for 'x': expected float32, got uint8"). So this variant
        # cannot be driven from a picture at all -- the example exists for the
        # postprocessor and the embedding contract, and the README carries the
        # open_clip snippet that produces the tokens.
        task="image_retrieval", task_legacy="embedding", family="clip_rn50",
        cpp_donor="image_retrieval/clip_rn50",
        variants=["clip-text_resnet50_77x512_openai"],
        preprocessor=SIMPLE,
        postprocessor=_proc("CLIPTextPostprocessor", WHC),
        visualizer=_proc("EmbeddingVisualizer", []),
        base="IEmbeddingFactory", reference=REF_CLIP, config={},
    ),
    dict(
        # Zoo category: Visual Place Recognition. It was filed under super_resolution
        # purely because the legacy task tables had nowhere else to put an embedding.
        task="visual_place_recognition", task_legacy="embedding",
        family="pp_shitu_rec",
        cpp_donor="visual_place_recognition/eigenplaces",
        variants=["pp-shituv2-feature-extraction_224x224"],
        preprocessor=SIMPLE,
        postprocessor=_proc("GalleryRetrievalPostprocessor", WHC,
                            model_type="place_descriptor",
                            model_name="pp-shituv2-feature-extraction_224x224"),
        visualizer=_proc("RetrievalVisualizer", [{"token": "config"}]),
        base="IEmbeddingFactory", reference=REF_PADDLECLAS, config={},
    ),
    dict(
        # Zoo category: Person ReID.
        task="person_reid", task_legacy="reid", family="repvgg_reid",
        cpp_donor="image_classification/casvit",
        variants=["repvgg-a0-reid_256x128"],
        preprocessor=SIMPLE,
        postprocessor=_proc("GalleryRetrievalPostprocessor", WHC,
                            model_type="person_reid_embedding",
                            model_name="repvgg-a0-reid_256x128"),
        visualizer=_proc("RetrievalVisualizer", [{"token": "config"}]),
        base="IEmbeddingFactory", reference=REF_PADDLECLAS, config={},
    ),

    # ============================ PP-ShiTu detection (2) ========================
    dict(
        # PP-PicoDet mainbody detector, one foreground class. NanoDetPostprocessor was
        # the first choice -- PicoDet and NanoDet-Plus share the GFL/DFL head -- and it
        # does not fit, for packaging rather than algorithm. MEASURED: this model emits
        # EIGHT tensors, a score and a box tensor per pyramid level
        # ((1,6400,1)...(1,100,32), strides 8/16/32/64, reg_max 7), where NanoDet emits
        # one concatenated (1, N, nc + 4*(reg_max+1)).
        task="mainbody_detection", task_legacy="object_detection", family="pp_shitu",
        cpp_donor="object_detection/nanodet",
        variants=["pp-shituv1-mainbody-detection_640x640",
                  "pp-shituv2-mainbody-detection_640x640"],
        preprocessor=LETTERBOX,
        postprocessor=_proc("PicoDetPostprocessor", WHC),
        visualizer=_proc("DetectionVisualizer", [], custom_labels=["mainbody"]),
        base="IDetectionFactory", reference=REF_PADDLECLAS,
        config={"num_classes": 1, "conf_threshold": 0.4, "nms_threshold": 0.5,
                "class_names": ["mainbody"]},
    ),

    # ============================ DarkPose (1) ==================================
    dict(
        # VitPosePostprocessor is a shape-agnostic heatmap argmax decoder: it reads
        # (num_keypoints, hm_h, hm_w) off the tensor, so HRNet-W32's [1, 17, 64, 48]
        # decodes without a new class. It uses the standard quarter-offset shift rather
        # than DarkPose's distribution-aware refinement, so keypoints land a fraction of
        # a heatmap cell less precisely than the published DarkPose metric.
        task="pose_estimation", task_legacy="pose_estimation", family="dark_hrnet",
        cpp_donor="pose_estimation/vitpose",
        variants=["dark-hrnet-w32_256x192"],
        preprocessor=SIMPLE,
        postprocessor=_proc("VitPosePostprocessor", WHC),
        visualizer=_proc("PoseVisualizer", []),
        base="IPoseFactory", reference=REF_HRNET,
        config={"num_keypoints": 17},
    ),
]

# The 4 zoo rows carrying a q-master .dxnn but no q-lite one. This decides the
# download TIER and nothing else -- it used to double as "is this published", which
# stopped being true when the zoo published 2_5_0. Measured 2026-09-22 and again
# 2026-09-30: HTTP 200 at modelzoo/q-master-dxnn/2_4_0/.
Q_MASTER_ONLY = frozenset({
    "beit-l-p16_384x384", "levit-128s_224x224",
    "vit-b-p16_384x384", "vit-t-p16_224x224",
})

# Declared but NOT downloadable. Empty since vit-l-p16_512x512_swag was published
# at q-lite-dxnn/2_5_0 (HTTP 200). A name in this set is registry published=false
# and a manifest row with pending=true.
#
# Do not infer unpublished from the version directory. That was correct only while
# 2_5_0 was unreleased; after publication it marks the whole q-lite tier pending.
UNPUBLISHED = frozenset()


def _legacy_model_name(variant: str) -> str:
    """A unique legacy-compatible key. Variants are unique, so this is too."""
    return variant.replace("-", "_").replace(".", "_")


def expand() -> list[dict]:
    """One fully resolved row per variant, in the shape the emitter consumes."""
    rows: list[dict] = []
    for group in GROUPS:
        for variant in group["variants"]:
            if variant not in ZOO_ROWS:
                raise KeyError(f"{variant} is not in the harvested zoo table")
            display_name, _, zoo_task, dataset, license_ = ZOO_ROWS[variant]
            width, height = resolution(variant)
            rows.append({
                "variant": variant,
                "task": group["task"],
                "task_legacy": group["task_legacy"],
                "family": group["family"],
                "width": width,
                "height": height,
                "display_name": display_name,
                "zoo_task": zoo_task,
                "dataset": dataset,
                "license": license_,
                "reference": group["reference"],
                "config": {**group["config"],
                           **(group.get("config_by_variant") or {}).get(variant, {})},
                "preprocessor": group["preprocessor"],
                "postprocessor": group["postprocessor"],
                "visualizer": group["visualizer"],
                "factory_bases": [group["base"]],
                # A pose factory must carry get_num_keypoints or it cannot be built.
                "extra_methods": group.get(
                    "extra_methods",
                    COCO_17_KEYPOINTS if group["base"] == "IPoseFactory" else []),
                "imports": [
                    f"from common.base import {group['base']}",
                    f"from common.processors import {group['preprocessor']['class']}, "
                    f"{group['postprocessor']['class']}",
                    f"from common.visualizers import {group['visualizer']['class']}",
                ],
                "cpp_donor": group["cpp_donor"],
                "legacy_model_name": _legacy_model_name(variant),
                "manifest_category": MANIFEST_CATEGORY[group["task_legacy"]],
                "published": variant not in UNPUBLISHED,
                # q-lite carries 492 of the 496 files the zoo page offers; the tier is
                # a property of the model, not of its publication state.
                "manifest_tier": ("q-master", "2_4_0")
                                 if variant in Q_MASTER_ONLY else ("q-lite", "2_5_0"),
            })
    return rows


def self_check() -> None:
    """Fail loudly on a table that cannot be what it claims to be."""
    rows = expand()
    variants = [r["variant"] for r in rows]
    duplicates = {v for v in variants if variants.count(v) > 1}
    if duplicates:
        raise AssertionError(f"variant declared twice: {sorted(duplicates)}")
    missing = set(ZOO_ROWS) - set(variants)
    if missing:
        raise AssertionError(f"zoo row never claimed by a group: {sorted(missing)}")
    if len(rows) != 147:
        raise AssertionError(f"expected 147 rows, got {len(rows)}")
    # One family must live in exactly one task: CMake target names are family-derived.
    owner: dict[str, str] = {}
    for r in rows:
        if owner.setdefault(r["family"], r["task"]) != r["task"]:
            raise AssertionError(
                f"family {r['family']} spans {owner[r['family']]} and {r['task']}")


if __name__ == "__main__":
    self_check()
    rows = expand()
    print(f"{len(rows)} rows, {len({r['family'] for r in rows})} families, "
          f"{len({r['task'] for r in rows})} tasks, "
          f"{sum(1 for r in rows if r['published'])} published")
