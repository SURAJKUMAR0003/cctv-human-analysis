# Datasets

Two independent datasets can live here. Neither is included in the repository:
you bring your own footage and annotations.

```
datasets/
├── yolo/            your YOLO detection / pose dataset
├── expression/      your facial-expression dataset
└── samples/         small clips used for offline testing (generated)
```

---

## 1. YOLO pose dataset

### Directory layout

Ultralytics finds labels by replacing `/images/` with `/labels/` in the image
path, so the two trees must mirror each other exactly.

```
datasets/yolo/
├── dataset.yaml
├── images/
│   ├── train/   cam01_000123.jpg, cam01_000124.jpg, ...
│   ├── val/
│   └── test/        (optional)
└── labels/
    ├── train/   cam01_000123.txt, cam01_000124.txt, ...
    ├── val/
    └── test/
```

Every image needs a `.txt` file with the same stem. An image with no people
gets an **empty** `.txt` file - not a missing one. Empty files are valid
negative examples and they matter: they teach the model what is not a person.

### Annotation format

All coordinates are normalised to `0..1` against the image width and height.

**Detection** (one line per object):

```
<class_id> <x_center> <y_center> <width> <height>
```

**Pose** (one line per person - what this project uses):

```
<class_id> <x_center> <y_center> <width> <height> <px1> <py1> <v1> <px2> <py2> <v2> ...
```

- `class_id` is `0` for person (the only class this pipeline consumes).
- 17 keypoint triplets follow, in the COCO order below.
- `v` is visibility: `0` not labelled, `1` labelled but occluded, `2` visible.
  With `kpt_shape: [17, 3]` you must provide all three values per keypoint.

Keypoint order (index: name) - this order is hard-wired into the API payload,
so a custom model must keep it:

```
0 nose          5 left_shoulder    11 left_hip
1 left_eye      6 right_shoulder   12 right_hip
2 right_eye     7 left_elbow       13 left_knee
3 left_ear      8 right_elbow      14 right_knee
4 right_ear     9 left_wrist       15 left_ankle
               10 right_wrist      16 right_ankle
```

Example line - one person, box centred at 45%/60% of the frame, 20% wide and
70% tall, with the nose visible and the left eye occluded:

```
0 0.450 0.600 0.200 0.700 0.452 0.310 2 0.461 0.302 1 ...
```

A short sanity rule: every value on the line is between 0 and 1 except
`class_id` and the visibility flags.

### dataset.yaml

```yaml
# datasets/yolo/dataset.yaml
path: /absolute/path/to/datasets/yolo   # dataset root (or relative to this file)
train: images/train
val: images/val
test: images/test        # optional

kpt_shape: [17, 3]       # 17 keypoints, (x, y, visibility)

# Left/right keypoint pairs, swapped when an image is mirrored during
# augmentation. Without this, horizontal flips corrupt your labels.
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
  0: person
```

For a plain detection dataset (no keypoints), drop `kpt_shape` and `flip_idx`
and train from a detection checkpoint such as `yolo11n.pt`.

### Train, validate, infer, export

```bash
# Train (the script refuses to start unless it finds real images and labels)
python scripts/train_yolo.py \
    --data datasets/yolo/dataset.yaml \
    --model yolo11n-pose.pt \
    --epochs 100 --imgsz 640 --batch 16 --device 0

# Validate a checkpoint
python scripts/validate_yolo.py \
    --weights runs/pose_custom/weights/best.pt \
    --data datasets/yolo/dataset.yaml

# Try it on images, a folder, a video or a camera
python scripts/infer_yolo.py \
    --weights runs/pose_custom/weights/best.pt \
    --source datasets/samples/bus.jpg --show-json

# Export (also available as --export onnx on train_yolo.py)
python scripts/train_yolo.py --data ... --export onnx
```

### Using your model in the live system

```bash
# .env
YOLO_MODEL=/absolute/path/to/runs/pose_custom/weights/best.pt
```

Restart the backend. No code changes. `.pt`, `.onnx`, `.engine` and
`.torchscript` all work, because the loader goes through Ultralytics.

### Practical notes

- Aim for at least a few thousand annotated people; a few hundred will
  overfit and you will see it in the val metrics, not in the demo.
- Annotate from the same cameras, angles and lighting you will deploy on.
  A model trained on web photos degrades sharply on ceiling-mounted CCTV.
- Keep the val split from *different* recordings than train. Splitting
  consecutive frames of one clip inflates every metric you will read.
- Labelling tools that export this format directly: CVAT, Label Studio,
  Roboflow.

---

## 2. Facial-expression dataset

Only needed if you want to replace the bundled expression model.

```
datasets/expression/
├── train/
│   ├── neutral/    img_0001.jpg ...
│   ├── happiness/
│   └── ...
└── val/
    └── ...
```

Train with any framework you like, then export to ONNX with a single image
input and one logits output. Drop the file into `backend/models/` and write a
sidecar JSON beside it describing labels and preprocessing:

```json
{
  "labels": ["neutral", "happiness", "sadness", "surprise"],
  "input_size": [224, 224],
  "color_order": "rgb",
  "scale": 0.00392156862745098,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "layout": "nchw",
  "apply_softmax": true
}
```

Then set `EXPRESSION_MODEL=your_model.onnx`. The loader validates that your
label count matches the model's output width and refuses to start if they
disagree, so a mislabelled output can't silently show wrong percentages.

If your model needs preprocessing this JSON can't express, implement
`ExpressionClassifier` (`backend/app/expression/base.py`) instead - the
interface is a single `predict(face_image) -> dict[str, float]`.

**Scope reminder:** an expression classifier reports how a face *looks* to a
model. It does not measure what a person feels, and no dataset you train on
will change that.

---

## 3. Sample clips

`datasets/samples/` holds test footage generated by:

```bash
python scripts/make_test_video.py
```

It pans and zooms over real photographs so the pipeline has genuine people,
faces and motion to work with on machines without a camera. The video is test
*input* - every detection drawn from it comes from the real models.
