import cv2
import numpy as np
import torch
import mediapipe as mp
from einops import rearrange
from typing import List, Union, Tuple, Dict, Optional
import os
import threading
from tqdm import tqdm
import logging
import traceback

class FaceAlignmentProcessor:
    """
    A class for processing videos, detecting facial landmarks,
    and performing face alignment using MediaPipe.
    """
    
    def __init__(self, 
                 resolution: int = 512, 
                 device: str = "cpu",
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the FaceAlignmentProcessor.
        
        Args:
            resolution: Output resolution for aligned faces.
            device: Device to run processing on ('cpu' or 'cuda:X').
            static_image_mode: Whether to treat each frame as a static image.
            max_num_faces: Maximum number of faces to detect.
            min_detection_confidence: Minimum confidence for face detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
        """
        self.resolution = resolution
        self.device = device
        
        # Initialize MediaPipe face mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize laplacian smoother and affine restorer
        self.smoother = laplacianSmooth()
        self.restorer = AlignRestore()
        
        # Mapping for landmark conversion
        self.landmark_points_68 = [
            162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 
            378, 365, 397, 288, 323, 454, 389, 71, 63, 105, 
            66, 107, 336, 296, 334, 293, 301, 168, 197, 5, 
            4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 
            144, 362, 385, 387, 263, 373, 380, 61, 39, 37, 
            0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 
            13, 312, 308, 317, 14, 87
        ]
        self.lock = threading.Lock()
        
    def detect_face(self,
                     image: Union[np.ndarray, torch.Tensor]) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the given image.
        """
        # Convert torch tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image = rearrange(image, "c h w -> h w c").numpy()
            else:
                image = image.numpy()
        elif isinstance(image, list):
            image = np.array(image)
            print(f"Image is a list, converting to numpy array of shape {image.shape}")
                
        # Make sure image is RGB (MediaPipe expects RGB)
        if image.shape[2] == 3 and (image.dtype == np.uint8):
            # If likely BGR (OpenCV default), convert to RGB
            if np.mean(image[:,:,0]) < np.mean(image[:,:,2]):
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                logging.warning("Image is likely in BGR format, please check")
                from matplotlib import pyplot as plt
                plt.imshow(image)
                plt.axis('off')
                plt.show()
                
        height, width = image.shape[:2]
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:  # Face not detected
            return []
        face_landmarks = results.multi_face_landmarks[0]  # Only use the first face
        # Extract bounding box coordinates
        x_min = int(min([lm.x for lm in face_landmarks.landmark]) * width)
        x_max = int(max([lm.x for lm in face_landmarks.landmark]) * width)
        y_min = int(min([lm.y for lm in face_landmarks.landmark]) * height)
        y_max = int(max([lm.y for lm in face_landmarks.landmark]) * height)
        # Return bounding box coordinates
        return [(x_min, y_min, x_max, y_max)]
    
    def detect_landmarks(self, 
                         image: Union[np.ndarray, torch.Tensor], 
                         return_2d: bool = True,
                         convert_to_fa_format: bool = False) -> Optional[np.ndarray]:
        """
        Detect facial landmarks in the given image.
        
        Args:
            image: Input image as numpy array or torch tensor.
            return_2d: Whether to return 2D landmarks only.
            convert_to_fa_format: Whether to convert to face_alignment format (68 points).
            
        Returns:
            Array of landmark coordinates or None if no face detected.
        """
        # Convert torch tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
                image = rearrange(image, "c h w -> h w c").numpy()
            else:
                image = image.numpy()
                
        # Make sure image is RGB (MediaPipe expects RGB)
        if image.shape[2] == 3 and (image.dtype == np.uint8):
            # If likely BGR (OpenCV default), convert to RGB
            if np.mean(image[:,:,0]) < np.mean(image[:,:,2]):  # Simple heuristic to detect BGR
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                logging.warning("Image is likely in BGR format, please check")
                from matplotlib import pyplot as plt
                plt.imshow(image)
                plt.axis('off')
                plt.show()
        
        height, width = image.shape[:2]
        results = self.face_mesh.process(image)
        
        if not results.multi_face_landmarks:  # Face not detected
            return None
            
        face_landmarks = results.multi_face_landmarks[0]  # Only use the first face
        
        # Extract landmark coordinates
        landmark_coordinates = []
        for landmark in face_landmarks.landmark:
            x = landmark.x * width
            y = landmark.y * height
            z = landmark.z * width  # Scale Z relative to image width
            
            if return_2d:
                landmark_coordinates.append((x, y))
            else:
                landmark_coordinates.append((x, y, z))
                
        landmark_coordinates = np.array(landmark_coordinates)
        
        # Convert to face_alignment format if requested
        if convert_to_fa_format:
            return self.convert_to_face_alignment_format(landmark_coordinates)
            
        return landmark_coordinates
    
    def convert_to_face_alignment_format(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Convert MediaPipe's 478 landmarks to face_alignment's 68 landmarks format.
        
        Args:
            landmarks: MediaPipe facial landmarks array [478, 2] or [478, 3]
            
        Returns:
            Array of 68 landmark points in face_alignment format.
        """
        landmarks_extracted = []
        for index in self.landmark_points_68:
            if index < len(landmarks):
                if landmarks.shape[1] == 2:
                    landmarks_extracted.append((landmarks[index][0], landmarks[index][1]))
                else:
                    landmarks_extracted.append((landmarks[index][0], landmarks[index][1]))
        
        return np.array(landmarks_extracted)
    
    def get_aligned_faces(self, 
                         image: Union[np.ndarray, torch.Tensor],
                         old_state: Optional[Dict] = {},
                         return_box: bool = False,
                         return_matrix: bool = False,
                         upscale_interpolation=cv2.INTER_CUBIC,
                         downscale_interpolation=cv2.INTER_AREA,
                        ) -> Union[np.ndarray, Tuple]:
        """
        Detect face and return aligned face.
        
        Args:
            image: Input image
            return_box: Whether to return the face bounding box
            return_matrix: Whether to return the affine transformation matrix
            
        Returns:
            Aligned face image, and optionally bounding box and affine matrix
        """
        image = image.copy()
        # Convert torch tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
                image = rearrange(image, "c h w -> h w c").numpy()
            else:
                image = image.numpy()
        
        # Ensure we have a contiguous copy of the image for modification
        # image_copy = image.copy()
        
        # Get landmarks in face_alignment format
        landmarks = self.detect_landmarks(image, return_2d=True, convert_to_fa_format=True)
        
        if landmarks is None:
            # print(f"No face detected in the image of length {len(image)}")
            # In this case, for debugging, lets plot the image
            # from matplotlib import pyplot as plt
            # plt.imshow(image)
            # plt.axis('off')
            # plt.show()
            raise RuntimeError("Face not detected")
        
        # Apply Laplacian smoothing to landmarks
        points, pts_last = self.smoother.smooth(landmarks, old_state.get("pts_last", None))
        
        # Calculate reference points for alignment
        lmk3_ = np.zeros((3, 2))
        lmk3_[0] = points[17:22].mean(0)  # Left eye region
        lmk3_[1] = points[22:27].mean(0)  # Right eye region
        lmk3_[2] = points[27:36].mean(0)  # Nose region
        
        # Align and warp the face
        face, affine_matrix, p_bias = self.restorer.align_warp_face(
            image, 
            p_bias=old_state.get("p_bias", None),
            lmks3=lmk3_, smooth=True, border_mode="constant"
        )
        
        # Get bounding box
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        
        # Calculate original dimensions
        H, W, C = image.shape
        is_downscaling = max(H, W) > self.resolution
        interpolation = downscale_interpolation if is_downscaling else upscale_interpolation
        
        # Resize to target resolution
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=interpolation)
        
        state = {
            "pts_last": pts_last,
            "p_bias": p_bias
        }
        
        # Determine what to return
        if return_box and return_matrix:
            return face, box, affine_matrix, state
        elif return_box:
            return face, box, state
        elif return_matrix:
            return face, affine_matrix, state
        else:
            return face, state
        
    def process_frames(self, frames: List[np.ndarray], break_on_error: bool = False) -> List[np.ndarray]:
        """
        Process a list of frames and extract aligned faces.
        
        Args:
            frames: List of frames (numpy arrays)
            
        Returns:
            List of aligned face images
        """
        with self.lock:
            aligned_faces = []
            state = {}
            for frame in frames:
                try:
                    # print(f"Processing frame of shape: {frame.shape}")
                    aligned_face, state = self.get_aligned_faces(frame, old_state=state)
                    # print(f"Aligned face shape: {aligned_face.shape}")
                    aligned_faces.append(aligned_face)
                except RuntimeError as e:
                    # print(f"Error processing frame: {e}. Only processed {len(aligned_faces)} frames.")
                    # traceback.print_exc()
                    if break_on_error:
                        break
                    continue
            return aligned_faces
        
    def detect_faces(self, frames: List[np.ndarray], break_on_error: bool = False) -> List[np.ndarray]:
        """
        Detect faces in a list of frames.
        
        Args:
            frames: List of frames (numpy arrays)
            
        Returns:
            List of bounding boxes for detected faces
        """
        with self.lock:
            face_boxes = []
            for frame in frames:
                try:
                    boxes = self.detect_face(frame)
                    face_boxes.append(boxes)
                except Exception as e:
                    # print(f"Error detecting faces: {e}. Only processed {len(face_boxes)} frames.")
                    # traceback.print_exc()
                    if break_on_error:
                        break
                    continue
            return face_boxes
    
    def reset(self):
        """
        Reset the processor state.
        """
        self.smoother.pts_last = None
        self.restorer.p_bias = None
    
    def close(self):
        """
        Release resources.
        """
        if self.face_mesh:
            self.face_mesh.close()

import numpy as np
import cv2

def transformation_from_points(points1, points0, smooth=True, p_bias=None):
    points2 = np.array(points0, dtype=np.float64)
    points1 = points1.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(np.matmul(points1.T, points2))
    R = (np.matmul(U, Vt)).T
    sR = (s2 / s1) * R
    T = c2.reshape(2, 1) - (s2 / s1) * np.matmul(R, c1.reshape(2, 1))
    M = np.concatenate((sR, T), axis=1)
    if smooth:
        bias = points2[2] - points1[2]
        if p_bias is None:
            p_bias = bias
        else:
            bias = p_bias * 0.2 + bias * 0.8
        p_bias = bias
        M[:, 2] = M[:, 2] + bias
    return M, p_bias

class AlignRestore(object):
    def __init__(self, align_points=3):
        if align_points == 3:
            self.upscale_factor = 1
            ratio = 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            # self.p_bias = None

    def process(self, img, p_bias, lmk_align=None, smooth=True, align_points=3):
        # Removed debug writes for efficiency
        aligned_face, affine_matrix, p_bias = self.align_warp_face(img, lmk_align, p_bias=p_bias, smooth=smooth)
        restored_img = self.restore_img(img, aligned_face, affine_matrix)
        return aligned_face, restored_img

    def align_warp_face(self, img, lmks3, p_bias, smooth=True, border_mode="constant"):
        affine_matrix, p_bias = transformation_from_points(lmks3, self.face_template, smooth, p_bias)
        border_mode = {"constant": cv2.BORDER_CONSTANT,
                       "reflect101": cv2.BORDER_REFLECT101,
                       "reflect": cv2.BORDER_REFLECT}.get(border_mode, cv2.BORDER_CONSTANT)
        cropped_face = cv2.warpAffine(
            img,
            affine_matrix,
            self.face_size,
            flags=cv2.INTER_LINEAR,  # faster interpolation
            borderMode=border_mode,
            borderValue=[127, 127, 127],
        )
        return cropped_face, affine_matrix, p_bias

    def align_warp_face2(self, img, landmark, border_mode="constant"):
        affine_matrix = cv2.estimateAffinePartial2D(landmark, self.face_template)[0]
        border_mode = {"constant": cv2.BORDER_CONSTANT,
                       "reflect101": cv2.BORDER_REFLECT101,
                       "reflect": cv2.BORDER_REFLECT}.get(border_mode, cv2.BORDER_CONSTANT)
        cropped_face = cv2.warpAffine(
            img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132)
        )
        return cropped_face, affine_matrix

    def restore_img(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        upsample_img = cv2.resize(input_img, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
        inverse_affine = cv2.invertAffineTransform(affine_matrix)
        inverse_affine *= self.upscale_factor
        extra_offset = 0.5 * self.upscale_factor if self.upscale_factor > 1 else 0
        inverse_affine[:, 2] += extra_offset
        inv_restored = cv2.warpAffine(face, inverse_affine, (w_up, h_up), flags=cv2.INTER_LINEAR)
        mask = np.ones((self.face_size[1], self.face_size[0]), dtype=np.float32)
        inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
        inv_mask_erosion = cv2.erode(
            inv_mask, np.ones((max(1, int(2 * self.upscale_factor)), max(1, int(2 * self.upscale_factor))), np.uint8)
        )
        pasted_face = inv_mask_erosion[:, :, None] * inv_restored
        total_face_area = np.sum(inv_mask_erosion)
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = max(1, w_edge * 2)
        inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = max(1, w_edge * 2)
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
        inv_soft_mask = inv_soft_mask[:, :, None]
        upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
        return upsample_img.astype(np.uint8)

class laplacianSmooth:
    def __init__(self, smoothAlpha=0.3):
        self.smoothAlpha = smoothAlpha
        # self.pts_last = None

    def smooth(self, pts_cur, pts_last=None):
        if pts_last is None:
            pts_last = pts_cur.copy()
            return pts_cur.copy(), pts_last
        x1 = np.min(pts_cur[:, 0])
        x2 = np.max(pts_cur[:, 0])
        width = x2 - x1
        pts_update = []
        for i in range(len(pts_cur)):
            x_new, y_new = pts_cur[i]
            x_old, y_old = pts_last[i]
            tmp = (x_new - x_old) ** 2 + (y_new - y_old) ** 2
            w = np.exp(-tmp / (width * self.smoothAlpha + 1e-6))
            x = x_old * w + x_new * (1 - w)
            y = y_old * w + y_new * (1 - w)
            pts_update.append([x, y])
        pts_update = np.array(pts_update)
        pts_last = pts_update.copy()
        return pts_update, pts_last

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = FaceAlignmentProcessor(resolution=512)
    
    # Process a list of videos
    video_paths = [
        "path/to/video1.mp4",
        "path/to/video2.mp4"
    ]
    
    # Process with frame saving
    results = processor.process_videos(
        video_paths=video_paths,
        output_dir="aligned_faces",
        max_frames_per_video=100,
        frame_step=5,
        save_frames=True
    )
    
    # Clean up
    processor.close()