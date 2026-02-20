import os
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor, pipeline, Sam2Processor, Sam2Model


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-base"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    del object_detector

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything Model 2 (SAM2) to generate segmentation masks based on images and bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam2.1-hiera-large"

    # Use SAM2 model and processor
    model = Sam2Model.from_pretrained(segmenter_id).to(device)
    # processor = AutoProcessor.from_pretrained("facebook/sam-vit-huge")
    processor = Sam2Processor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = model(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    del model, processor

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections

def segment_video(
    video_path: str,
    labels: Optional[List[str]] = None,
    first_frame_masks: Optional[Dict[int, np.ndarray]] = None,
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Use SAM2 Video model to segment and track multiple objects in a video.

    Args:
        video_path (str): Path to the video file or URL
        labels (Optional[List[str]]): List of object labels to detect (used when first_frame_masks is not provided)
        first_frame_masks (Optional[Dict[int, np.ndarray]]): Dictionary of first frame masks, keys are object IDs, values are mask arrays
            If provided, the detection step will be skipped and these masks will be used directly
        threshold (float): Detection confidence threshold (only effective when using automatic detection)
        polygon_refinement (bool): Whether to use polygon refinement for masks
        detector_id (Optional[str]): Detector model ID
        segmenter_id (Optional[str]): Segmenter model ID, defaults to SAM2 Video model

    Returns:
        Dict[int, Dict[str, Any]]: Dictionary keyed by object ID, containing label and frames information
    """
    # Import video processing libraries
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    from transformers.video_utils import load_video
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load video frames
    video_frames, _ = load_video(video_path)
    first_frame = Image.fromarray(video_frames[0])
    
    # Determine processing flow based on whether first_frame_masks is provided
    if first_frame_masks is not None:
        # Use provided masks
        video_segments = {}
        for obj_idx, mask in first_frame_masks.items():
            video_segments[obj_idx] = {
                "label": f"object_{obj_idx}",  # Default label
                "frames": {0: mask}
            }
    else:
        # Use Grounding DINO to detect objects in the first frame
        if labels is None:
            raise ValueError("Either labels or first_frame_masks must be provided")
        
        detections = detect(first_frame, labels, threshold, detector_id)
        
        # If no objects detected, return empty result
        if not detections:
            return {}
        
        video_segments = {}
    
    # Load SAM2 Video model and processor
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam2.1-hiera-tiny"
    model = Sam2VideoModel.from_pretrained(segmenter_id, local_files_only=True).to(device, dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained(segmenter_id, local_files_only=True)
    
    # Initialize video inference session
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        dtype=torch.bfloat16,
    )
    
    # Add inputs for each object
    if first_frame_masks is not None:
        # Use provided masks as input
        for obj_idx, mask in first_frame_masks.items():
            # Convert mask to boolean type
            mask_bool = (mask > 0).astype(bool)
            
            # Add mask input to the first frame
            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=0,
                obj_ids=obj_idx,
                input_masks=torch.from_numpy(mask_bool[None, None, :, :]).to(device),
            )
            
            # Segment the object on the first frame
            outputs = model(
                inference_session=inference_session,
                frame_idx=0,
            )
            
            # Apply polygon refinement (if needed)
            if polygon_refinement:
                shape = mask.shape
                polygon = mask_to_polygon(mask)
                video_segments[obj_idx]["frames"][0] = polygon_to_mask(polygon, shape)
    else:
        # Use detection boxes as input
        for obj_idx, detection in enumerate(detections, start=1):
            box = detection.box.xyxy
            
            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=0,
                obj_ids=obj_idx,
                input_boxes=[[box]],
            )
            
            outputs = model(
                inference_session=inference_session,
                frame_idx=0,
            )
            
            # Process the segmentation result of the first frame
            frame_masks = processor.post_process_masks(
                [outputs.pred_masks], 
                original_sizes=[[inference_session.video_height, inference_session.video_width]], 
                binarize=True
            )[0]
            
            # Record the object's label and mask
            video_segments[obj_idx] = {
                "label": detection.label,
                "frames": {}
            }
            
            # Store the first frame's mask
            mask_np = frame_masks[0, 0].cpu().numpy().astype(np.uint8) * 255
            if polygon_refinement:
                shape = mask_np.shape
                polygon = mask_to_polygon(mask_np)
                mask_np = polygon_to_mask(polygon, shape)
            
            video_segments[obj_idx]["frames"][0] = mask_np
    
    # Use SAM2 Video model's propagation to track objects across all frames
    for frame_idx, sam2_video_output in enumerate(model.propagate_in_video_iterator(inference_session)):
        frame_masks = processor.post_process_masks(
            [sam2_video_output.pred_masks], 
            original_sizes=[[inference_session.video_height, inference_session.video_width]], 
            binarize=True
        )[0]
        
        # Process each object's mask
        for obj_idx in range(1, inference_session.get_obj_num() + 1):
            mask_np = frame_masks[obj_idx-1, 0].cpu().numpy().astype(np.uint8) * 255
            
            if polygon_refinement:
                shape = mask_np.shape
                polygon = mask_to_polygon(mask_np)
                mask_np = polygon_to_mask(polygon, shape)
            
            video_segments[obj_idx]["frames"][frame_idx] = mask_np
    
    # Clean up resources
    del model, processor, inference_session
    
    return video_segments

def visualize_video_segments(
    video_path: str, 
    video_segments: Dict[int, Dict[str, Any]], 
    output_path: str
) -> None:
    """
    Visualize video segmentation results and save as a new video

    Args:
        video_path (str): Path to the original video
        video_segments (Dict): Segmentation results
        output_path (str): Output video path
    """
    import cv2
    from transformers.video_utils import load_video
    
    # Load original video
    video_frames, info = load_video(video_path)
    fps = info.get("fps", 30)
    
    # Create video writer
    height, width = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Pre-allocate fixed colors for each object before processing video frames
    obj_colors = {}
    for obj_idx in video_segments.keys():
        obj_colors[obj_idx] = [random.randint(0, 255) for _ in range(3)]

    # Draw masks for each frame
    for frame_idx, frame in enumerate(video_frames):
        # Copy frame to avoid modifying original data
        vis_frame = frame.copy()
        
        # Draw mask for each object
        for obj_idx, obj_data in video_segments.items():
            if frame_idx in obj_data["frames"]:
                # Get the mask for the current frame
                mask = obj_data["frames"][frame_idx]
                
                # Use pre-allocated fixed color
                color = obj_colors[obj_idx]
                
                # Apply mask as a semi-transparent overlay
                overlay = vis_frame.copy()
                overlay[mask > 0] = color
                vis_frame = cv2.addWeighted(overlay, 0.5, vis_frame, 0.5, 0)
                
                # Add label
                label = obj_data["label"]
                cv2.putText(vis_frame, label, (10, 30 * (obj_idx+1)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Write frame
        out.write(vis_frame)
    
    # Release resources
    out.release()
    print(f"Visualization video saved to: {output_path}")

def run_img():
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    labels = ["a cat.", "a remote control."]
    threshold = 0.4

    detector_id = "IDEA-Research/grounding-dino-base"
    segmenter_id = "facebook/sam2.1-hiera-large"  # Use SAM2 model

    image_array, detections = grounded_segmentation(
        image=image_url,
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )

    print(image_array, detections)

def run_video():
    # Video segmentation example
    video_path = "benchmark_new/reference_videos/animal/bear_crop/3.mp4"  # or URL
    labels = ["bear"]
    threshold = 0.4
    
    detector_id = "IDEA-Research/grounding-dino-base"
    segmenter_id = "facebook/sam2.1-hiera-large"  # Use a smaller model for faster processing
    
    # Segment video
    video_segments = segment_video(
        video_path=video_path,
        labels=labels,
        threshold=threshold,
        polygon_refinement=False,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )
    
    # Visualize results
    visualize_video_segments(
        video_path=video_path,
        video_segments=video_segments,
        output_path="output_segmented_video.mp4"
    )

if __name__ == "__main__":
    # run_img()
    run_video()
    