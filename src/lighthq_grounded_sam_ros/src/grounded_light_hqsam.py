import os
import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

import prova_pkg_grounded_sam.groundingdino as GroundingDINO
import prova_pkg_grounded_sam.segment_anything as SAM
import prova_pkg_grounded_sam.LightHQSAM as LightHQSAM

from prova_pkg_grounded_sam.groundingdino.util.inference import Model
from prova_pkg_grounded_sam.segment_anything import SamPredictor
from prova_pkg_grounded_sam.LightHQSAM.setup_light_hqsam import setup_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_FOLDER = os.path.dirname(os.path.abspath(GroundingDINO.__file__))
GROUNDING_DINO_CONFIG_PATH = os.path.join(GROUNDING_DINO_FOLDER, "config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GROUNDING_DINO_FOLDER, "config/groundingdino_swint_ogc.pth")

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building MobileSAM predictor
HQSAM_FOLDER = os.path.dirname(os.path.abspath(LightHQSAM.__file__))
HQSAM_CHECKPOINT_PATH = os.path.join(HQSAM_FOLDER, "config/sam_hq_vit_tiny.pth")

checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
light_hqsam = setup_model()
light_hqsam.load_state_dict(checkpoint, strict=True)
light_hqsam.to(device=DEVICE)

sam_predictor = SamPredictor(light_hqsam)


# Predict classes and hyper-param for GroundingDINO
SOURCE_IMAGE_PATH = "/home/humans/base_seq_opt_multi_dynamic_env_SIM/images/lab_resized.jpg"
CLASSES = ["black robot that can move on the floor", "person"]
BOX_THRESHOLD = 0.5
TEXT_THRESHOLD = 0.5
NMS_THRESHOLD = 0.8


# load image
image = cv2.imread(SOURCE_IMAGE_PATH)

# detect objects
detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=CLASSES,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for confidence, class_id
        in zip(detections.confidence, detections.class_id)
    ]
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# save the annotated grounding dino image
# cv2.imwrite("./groundingdino_annotated_image.jpg", annotated_frame)
cv2.imshow("Annoted Image", annotated_frame)
# cv2.waitKey(0)

# NMS post process
print(f"Before NMS: {len(detections.xyxy)} boxes")
nms_idx = torchvision.ops.nms(
    torch.from_numpy(detections.xyxy), 
    torch.from_numpy(detections.confidence), 
    NMS_THRESHOLD
).numpy().tolist()

detections.xyxy = detections.xyxy[nms_idx]
detections.confidence = detections.confidence[nms_idx]
detections.class_id = detections.class_id[nms_idx]

print(f"After NMS: {len(detections.xyxy)} boxes")

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=False,
            hq_token_only=True,
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# convert detections to masks
detections.mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=detections.xyxy
)

# Build and visualize per-class binary masks (white = segmented area)
height, width = image.shape[:2]
for class_idx, class_name in enumerate(CLASSES):
    # gather all masks that belong to the current class
    class_mask_indices = np.where(detections.class_id == class_idx)[0]
    if len(class_mask_indices) == 0:
        continue

    # combine masks for this class into a single binary image
    combined_mask = np.any(detections.mask[class_mask_indices], axis=0)
    binary_image = (combined_mask.astype(np.uint8) * 255)

    # visualize; close window to continue
    cv2.imshow(f"Binary Mask - {class_name}", binary_image)
    # cv2.waitKey(0)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for confidence, class_id
        in zip(detections.confidence, detections.class_id)
    ]
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# save the annotated grounded-sam image
# cv2.imwrite("./grounded_light_hqsam_annotated_image.jpg", annotated_image)
cv2.imshow("Image B", annotated_image)
cv2.waitKey(0)
