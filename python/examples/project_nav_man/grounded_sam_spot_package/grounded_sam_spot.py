import cv2
import numpy as np
import supervision as sv
import argparse
import torch
import torchvision
from os import path

from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from MobileSAM.setup_mobile_sam import setup_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--MOBILE_SAM_CHECKPOINT_PATH", type=str, default="./models/mobile_sam.pt", help="model"
    )
    parser.add_argument(
        "--SOURCE_IMAGE_PATH", type=str, default="EfficientSAM/test_imgs/kitchen.jpg", help="path to image file"
    )
    parser.add_argument(
        "--OUT_FILE_BOX", type=str, default="EfficientSAM/test_output/groundingdino_annotated_image.jpg", help="the output filename"
    )
    parser.add_argument(
        "--OUT_FILE_SEG", type=str, default="EfficientSAM/test_output/grounded_mobile_sam_annotated_image.jpg", help="the output filename"
    )
    parser.add_argument(
        "--OUT_FILE_BIN_MASK", type=str, default="EfficientSAM/test_output/grounded_mobile_sam_bin_mask.jpg", help="the output filename"
    )
    parser.add_argument("--BOX_THRESHOLD", type=float, default=0.25, help="")
    parser.add_argument("--TEXT_THRESHOLD", type=float, default=0.25, help="")
    parser.add_argument("--NMS_THRESHOLD", type=float, default=0.8, help="")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--DEVICE", type=str, default='cpu', help="cuda:[0,1,2,3,4] or cpu"
    )
    return parser.parse_args()

def main(args):
  DEVICE = args.DEVICE
  dir_path = path.dirname(path.realpath(__file__))

  # GroundingDINO config and checkpoint
  GROUNDING_DINO_CONFIG_PATH = path.join(dir_path, "grounded_DINO/GroundingDINO_SwinT_OGC.py")
  GROUNDING_DINO_CHECKPOINT_PATH = path.join(dir_path, "models/groundingdino_swint_ogc.pth")

  # Building GroundingDINO inference model
  grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,device=DEVICE)

  # Building MobileSAM predictor
  MOBILE_SAM_CHECKPOINT_PATH = path.join(dir_path,args.MOBILE_SAM_CHECKPOINT_PATH)
  checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
  mobile_sam = setup_model()
  mobile_sam.load_state_dict(checkpoint, strict=True)
  mobile_sam.to(device=DEVICE)

  sam_predictor = SamPredictor(mobile_sam)


  # Predict classes and hyper-param for GroundingDINO
  SOURCE_IMAGE_PATH = args.SOURCE_IMAGE_PATH
  BOX_THRESHOLD = args.BOX_THRESHOLD
  TEXT_THRESHOLD = args.TEXT_THRESHOLD
  NMS_THRESHOLD = args.NMS_THRESHOLD
  
  CLASSES= [
    "Unknown", "Floor", "Door", "Countertop",
    "Open Shelf", "Bottle", "Can", "Storage", "Table", "Chair", 
    "Couch","Door Handle"
]

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
        f"{CLASSES[class_id_item]} {confidence_item:0.2f}" 
        for confidence_item, class_id_item in zip(detections.confidence, detections.class_id)
    ]
  annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

  # save the annotated grounding dino image
#   cv2.imwrite(args.OUT_FILE_BOX, annotated_frame)
#   print(args.OUT_FILE_BOX)


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
              multimask_output=True
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

  binary_mask = detections.mask[0].astype(np.uint8)*255
  cv2.imwrite(args.OUT_FILE_BIN_MASK, binary_mask)

  # annotate image with detections
  box_annotator = sv.BoxAnnotator()
  mask_annotator = sv.MaskAnnotator()
#   labels = [
#       f"{CLASSES[class_id]} {confidence:0.2f}" 
#       for _, _, confidence, class_id, _ 
#       in detections]
  labels = [
        f"{CLASSES[class_id_item]} {confidence_item:0.2f}" 
        for confidence_item, class_id_item in zip(detections.confidence, detections.class_id)
    ]
  annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
  annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
  # save the annotated grounded-sam image
  cv2.imwrite(args.OUT_FILE_SEG, annotated_image)
  
if __name__ == "__main__":
  args = parse_args()
  main(args)
