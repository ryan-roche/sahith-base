import cv2
import numpy as np
import supervision as sv
import argparse
import torch
import torchvision
from os import path
import sys

from groundingdino.util.inference import Model
from segment_anything import SamPredictor
current_directory = path.dirname(path.abspath(__file__))
mobilesam_path = path.join(current_directory, 'MobileSAM')
sys.path.append("/home/vsi/repo/robotdev/spot/spot-sdk/python/examples/sahith_test/grounded_sam_spot_package/MobileSAM")
if mobilesam_path not in sys.path:
    print("yes")
    sys.path.append(mobilesam_path)
from .MobileSAM.setup_mobile_sam import setup_model

DEVICE="cpu"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8
CLASSES= [
"Unknown", "Floor", "Door","Handle",
"Cabinet", "Bottle", "Can", "Table"
]

def segment_image(image,input_classes=CLASSES):

    if(len(image.shape)<3):
        image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    dir_path = path.dirname(path.realpath(__file__))

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = path.join(dir_path, "grounded_DINO/GroundingDINO_SwinT_OGC.py")
    GROUNDING_DINO_CHECKPOINT_PATH = path.join(dir_path, "models/groundingdino_swint_ogc.pth")

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,device=DEVICE)

    # Building MobileSAM predictor
    MOBILE_SAM_CHECKPOINT_PATH = path.join(dir_path,"./models/mobile_sam.pt")
    checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
    mobile_sam = setup_model()
    mobile_sam.load_state_dict(checkpoint, strict=True)
    mobile_sam.to(device=DEVICE)

    sam_predictor = SamPredictor(mobile_sam)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=input_classes,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{input_classes[class_id_item]} {confidence_item:0.2f}" 
        for confidence_item, class_id_item in zip(detections.confidence, detections.class_id)
    ]
    #annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

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
    # box_annotator = sv.BoxAnnotator()
    # annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    # cv2.imwrite("box_out.jpg", annotated_image)
    # cv2.imshow("detections",annotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # return detections

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
    #cv2.imwrite(args.OUT_FILE_BIN_MASK, binary_mask)

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    #   labels = [
    #       f"{input_classes[class_id]} {confidence:0.2f}" 
    #       for _, _, confidence, class_id, _ 
    #       in detections]
    labels = [
        f"{input_classes[class_id_item]} {confidence_item:0.2f}" 
        for confidence_item, class_id_item in zip(detections.confidence, detections.class_id)
    ]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    # save the annotated grounded-sam image
    cv2.imwrite("seg_out.jpg", annotated_image)
    cv2.imshow("segmentations",annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return detections

    # return detections

#test
if __name__=="__main__":
    img=cv2.imread("/home/vsi/repo/robotdev/spot/spot-sdk/python/examples/sahith_test/spot_img.jpg")
    segment_image(img,[
                "Unknown", "Floor", "Door", "Countertop",
                "Open Shelf", "Bottle", "Can", "Storage", "Table", "Chair", 
                "Couch","Door Handle"
                ])
