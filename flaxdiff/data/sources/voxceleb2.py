from logging import warn, warning
import os
import random
from arrow import get
import einops
import numpy as np
from os.path import join
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import decord
from decord import VideoReader, AudioReader, cpu
import traceback

from d2lv2_lightning.config import DataConfig
from d2lv2_lightning.utils import dist_util
from .face_mask import FaceMaskGenerator
from .prompt_templates import TEMPLATE_MAP
from .utils import ImageProcessor
from .audio import crop_wav_window, melspectrogram, crop_mel_window, get_segmented_wavs, get_segmented_mels

class Voxceleb2Decord(Dataset):
    """
    A dataset module for video-to-video (audio guided) diffusion training.
    This implementation uses decord to load videos and audio on the fly
    """
    default_video_fps = 25
    default_mel_steps_per_sec = 80.

    def __init__(
        self,
        split,
        data_config: DataConfig,  # expects attributes like: data_root, filelists_path, nframes, syncnet_mel_step_size, image_size, face_hide_percentage, video_fps, etc.
        tokenizer = None,
        token_map: dict = None,
        use_template: str = None,
        audio_format: str = "mel",
        h_flip: bool = True,
        color_jitter: bool = False,
        blur_amount: int = 70,
        sample_rate: int = 16000,
        shared_audio_dict=None,
        val_ratio: float = 0.001,
        num_val_ids: int = -1,
        val_split_seed: int = 787,
        dataset_name: str = "voxceleb2",
        face_mask_type: str = "fixed",
    ):
        random.seed(dist_util.get_rank() + 1)
        print(f"Dataset split: {split}, rank: {dist_util.get_rank() + 1}")
        self.split = split
        self.data_config = data_config
        self.tokenizer = tokenizer
        self.token_map = token_map
        self.use_template = use_template
        self.audio_format = audio_format
        self.h_flip = h_flip
        self.color_jitter = color_jitter
        self.blur_amount = blur_amount
        self.sample_rate = sample_rate
        self.shared_audio_dict = shared_audio_dict if shared_audio_dict is not None else {}
        self.val_ratio = val_ratio
        self.num_val_ids = num_val_ids
        self.val_split_seed = val_split_seed
        self.dataset_name = dataset_name
        self.face_mask_type = face_mask_type

        decord.bridge.set_bridge('torch')

        # Video properties (either from args or defaults)
        self.video_fps = getattr(data_config, "video_fps", self.__class__.default_video_fps)
        self.mel_steps_per_sec = self.__class__.default_mel_steps_per_sec

        # Set the data root based on the split.
        if split in ["train", "trainfull"]:
            self.data_root = os.path.join(data_config.data_root, "train")
        else:
            self.data_root = os.path.join(data_config.data_root, "test")
        # self.data_root = data_config.data_root

        # Determine file list path
        if hasattr(data_config, "filelists_path") and data_config.filelists_path is not None:
            self.filelists_path = data_config.filelists_path
        else:
            self.filelists_path = os.path.join('./data/voxceleb2/', "filelists")
            # Warn the user that the default filelists path is being used.
            warning(f"Using default filelists path: {self.filelists_path}. Please set data_config.filelists_path to a custom path if needed.")
        os.makedirs(self.filelists_path, exist_ok=True)

        filelist_file = join(self.filelists_path, f"{dataset_name}_{split}.txt")
        if not os.path.exists(filelist_file):
            warning(f"File list {filelist_file} not found. Creating a new file list. Please make sure to the data_root: {data_config.data_root} is correct for the split {split}.")
            self.all_videos = self.create_filelist()
        else:
            self.all_videos = self.get_video_list(filelist_file)
        print(f"Using file list: {filelist_file} with {len(self.all_videos)} videos.")

        # Image transforms (assumes 3-channel images)
        size = data_config.resolution
        self.size = size
        self.image_transforms = ImageProcessor(size)
        self.mask_transforms = ImageProcessor(size)
        
        if use_template is not None:
            assert token_map is not None, "token_map must be provided if using a template."
            self.templates = TEMPLATE_MAP[use_template]

    def worker_init_fn(self, worker_id):
        self.worker_id = worker_id
        if self.face_mask_type != "fixed":
            # Initialize dynamic face mask generator.
            self.mask_generator = FaceMaskGenerator(
                video_mode=False,
                mask_type=self.face_mask_type,
            )

        
    def get_video_list(self, filelist_file):
        videos = []
        with open(filelist_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    # Each line is relative to data_root.
                    videos.append(os.path.join(self.data_root, line))
        return videos

    def create_filelist(self):
        # Create a filelist by scanning the directory structure.
        # (This example assumes VoxCeleb2 videos are stored under data_root/id/vid/utterance.mp4)
        all_videos = []
        print("Creating filelist for dataset", self.dataset_name)
        if self.dataset_name == 'voxceleb2':
            for identity in os.listdir(self.data_root):
                id_path = os.path.join(self.data_root, identity)
                if not os.path.isdir(id_path):
                    continue
                for vid in os.listdir(id_path):
                    vid_path = os.path.join(id_path, vid)
                    if not os.path.isdir(vid_path):
                        continue
                    for utt in os.listdir(vid_path):
                        if utt.endswith(".mp4") or utt.endswith(".avi"):
                            # Save relative path (so that data_root can be prepended)
                            all_videos.append(os.path.join(identity, vid, utt))
        else:
            raise NotImplementedError("Filelist creation for this dataset is not implemented.")
        print("Total videos found:", len(all_videos))
        # Write filelist to disk.
        filelist_file = join(self.filelists_path, f"{self.dataset_name}_{self.split}.txt")
        with open(filelist_file, "w") as f:
            for v in all_videos:
                f.write(v + "\n")
        # Return full paths.
        return [os.path.join(self.data_root, v) for v in all_videos]

    def get_masks(self, imgs, pad=0):
        if hasattr(self, 'mask_generator'):
            try:
                if imgs.shape[-1] == 3:
                    B, H, W, C = imgs.shape
                else:
                    B, C, H, W = imgs.shape
                    imgs = einops.rearrange(imgs, "b c h w -> b h w c")
                masks = self.mask_generator.generate_mask_video(imgs.numpy(), mask_expansion=10, expansion_factor=1.1)
                return torch.from_numpy(np.stack(masks, axis=0, dtype=np.float16).reshape(B, 1, H, W) // 255)
            except Exception as e:
                print(f"Error generating masks with mask_generator: {e}")
                # Fallback to simple mask generation if the generator fails.
                print("Falling back to simple mask generation.")
                return self.get_simple_mask(pad)
        else:
            return self.get_simple_mask(pad)

    def get_simple_mask(self, pad=0):
        if getattr(self, 'mask_cache', None) is not None:
            return self.mask_cache
        H = W = self.size
        # Define a crop region similar to the original crop function.
        y1, y2 = 0, H - int(H * 2.36 / 8)
        x1, x2 = int(W * 1.8 / 8), W - int(W * 1.8 / 8)
        # Apply face_hide_percentage to determine the mask region.
        y1 = y2 - int(np.ceil(self.data_config.face_hide_percentage * (y2 - y1)))
        if pad:
            y1 = max(y1 - pad, 0)
            y2 = min(y2 + pad, H)
            x1 = max(x1 - pad, 0)
            x2 = min(x2 + pad, W)
        msk = Image.new("L", (W, H), 0)
        msk_arr = np.array(msk).astype(np.float16)
        msk_arr[y1:y2, x1:x2] = 255
        
        msk_arr = msk_arr // 255
        
        # msk = Image.fromarray(msk_arr)
        # msk = self.mask_transforms.preprocess_frames(msk) * 0.5 + 0.5  # normalize to [0,1]
        # Duplicate the mask for each frame.
        mask = torch.from_numpy(msk_arr).to(torch.float16).unsqueeze(0).repeat(self.data_config.nframes, 1, 1, 1)
        # Cache the mask for all frames.
        self.mask_cache = mask
        return mask

    def read_frames(self, videoreader: VideoReader, start_frame, num_frames):
        """
        Read a batch of frames from the video using decord.
        Returns a tuple: (list of transformed frames, list of reference frames, list of raw PIL frames).
        """
        try:
            total_frames = len(videoreader)
            if total_frames < num_frames:
                return None, None, None
            # Get the target window of frames.
            frame_indices = list(range(start_frame, start_frame + num_frames))
            frames_array = videoreader.get_batch(frame_indices)  # shape: (num_frames, H, W, C)
            
            # Determine valid start indices for a "wrong" window that does not overlap the instance window.
            valid_starts = []
            # Left interval: ensure wrong_start + num_frames - 1 < start_frame.
            left_max = start_frame - num_frames
            if left_max >= 0:
                valid_starts.extend(range(0, left_max + 1))
            # Right interval: ensure wrong_start > start_frame + num_frames - 1.
            right_min = start_frame + num_frames
            if right_min <= total_frames - num_frames:
                valid_starts.extend(range(right_min, total_frames - num_frames + 1))

            if not valid_starts:
                # Fallback: if no valid index is available, choose the farthest possible window.
                wrong_start = 0 if start_frame > total_frames // 2 else total_frames - num_frames
            else:
                wrong_start = random.choice(valid_starts)

            wrong_indices = list(range(wrong_start, wrong_start + num_frames))
                
            wrong_indices = list(range(wrong_start, wrong_start + num_frames))
            wrong_array = videoreader.get_batch(wrong_indices)
            return frames_array, wrong_array
        except Exception as e:
            print(f"Error reading frames from {videoreader}: {e}")
            return None, None, None

    def read_audio(self, video_path):
        try:
            ar = AudioReader(video_path, ctx=cpu(self.worker_id), sample_rate=self.sample_rate)
            audio = ar[:].squeeze()  # assume mono
            del ar
            return audio
        except Exception as e:
            print(f"Error reading audio from {video_path}: {e}")
            return None

    def compute_mel(self, audio):
        try:
            mel = melspectrogram(audio)
            return mel.T
        except Exception as e:
            print("Error computing mel spectrogram:", e)
            return None
        
    def get_mel(self, audio, path):
        # First try to find the mel in the cache directory
        cache_dir = self.data_config.data_cache_path if self.data_config.data_cache_path else os.path.join(self.data_root, "cache")
        cache_dir = os.path.join(cache_dir, self.split)
        cache_path = os.path.join(cache_dir, os.path.basename(path) + ".mel")
        if os.path.exists(cache_path):
            mel = np.load(cache_path)
            return mel
        # If not found, compute the mel and save it to the cache
        mel = self.compute_mel(audio)
        if mel is None:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, mel)
        return mel

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, index):
        """
        Returns a dictionary with:
          - instance_images: [F, C, H, W]
          - reference_images: [F, C, H, W]
          - mask: [F, 1, H, W]
          - instance_masks: same as mask
          - (optionally) instance_masks_dilated
          - instance_masked_images: instance_images * (mask < 0.5)
          - instance_prompt_ids: tokenized caption
          - raw_audio / indiv_raw_audios, mels / indiv_mels if audio_format is specified.
        """
        example = {}
        attempt = 0
        while True:
            attempt += 1
            if attempt > 10:
                raise RuntimeError("Failed to get a valid sample after multiple attempts.")
            try:
                # Select a random video.
                video_idx = random.randint(0, len(self.all_videos) - 1)
                video_path = self.all_videos[video_idx]
                vr = VideoReader(video_path, ctx=cpu(self.worker_id))
                total_frames = len(vr)
                if total_frames < 3 * self.data_config.nframes:
                    continue

                # Randomly choose a start frame ensuring enough frames for the window.
                start_frame = random.randint(self.data_config.nframes // 2, total_frames - self.data_config.nframes - self.data_config.nframes // 2)
                inst_frames, ref_frames = self.read_frames(vr, start_frame, self.data_config.nframes)
                if inst_frames is None or ref_frames is None:
                    continue
                
                vr.seek(0)  # avoid memory leak
                del vr    
                
                # Generate masks
                masks = self.get_masks(inst_frames)
                masks = self.image_transforms.resize(masks)
                
                dilated_masks = None
                if getattr(self.data_config, "dilate_masked_loss", False):
                    dilated_masks = self.get_masks(inst_frames, pad=self.data_config.resolution // 10)
                    dilated_masks = self.image_transforms.resize(dilated_masks)
                
                # Preprocess frames.
                inst_frames = self.image_transforms.preprocess_frames(inst_frames)
                ref_frames = self.image_transforms.preprocess_frames(ref_frames)

                # Optionally apply horizontal flip.
                if self.h_flip and random.random() > 0.5:
                    inst_frames = F.hflip(inst_frames)
                    ref_frames = F.hflip(ref_frames)
                    masks = F.hflip(masks)
                    if dilated_masks is not None:
                        dilated_masks = F.hflip(dilated_masks)
                    
                # Audio processing.
                if "wav" in self.audio_format or "mel" in self.audio_format:
                    audio = self.read_audio(video_path)

                    audio_chunk = crop_wav_window(
                        audio, 
                        start_frame=start_frame,
                        nframes=self.data_config.nframes,
                        video_fps=self.video_fps,
                        sample_rate=self.sample_rate,
                    )
                    if audio_chunk is None:
                        continue
                    example["raw_audio"] = audio_chunk
                    if getattr(self.data_config, "use_indiv_audio", False):
                        indiv_audios = get_segmented_wavs(
                            audio, 
                            start_frame, 
                            self.data_config.nframes, 
                            self.video_fps, 
                            self.sample_rate, 
                            indiv_audio_mode=self.data_config.indiv_audio_mode,
                        )
                        example["indiv_raw_audios"] = torch.FloatTensor(indiv_audios)
                    if "mel" in self.audio_format:
                        mel = self.get_mel(audio, video_path)
                        if mel is None:
                            continue
                        mel_window = crop_mel_window(
                            mel,
                            start_frame,
                            self.mel_steps_per_sec,
                            self.data_config.syncnet_mel_step_size,
                            self.video_fps,
                        )
                        if mel_window.shape[0] != self.data_config.syncnet_mel_step_size:
                            continue
                        example["mels"] = torch.FloatTensor(mel_window.T).unsqueeze(0)
                        indiv_mels = get_segmented_mels(
                            mel,
                            start_frame,
                            self.data_config.nframes,
                            self.mel_steps_per_sec,
                            self.data_config.syncnet_mel_step_size,
                            self.video_fps,
                        )
                        if indiv_mels is None:
                            continue
                        example["indiv_mels"] = torch.FloatTensor(indiv_mels)

                example["instance_images"] = inst_frames  # [F, C, H, W]
                example["reference_images"] = ref_frames  # [F, C, H, W]
                example["mask"] = masks  # [F, 1, H, W]
                example["instance_masks"] = example["mask"]
                if dilated_masks is not None:
                    example["instance_masks_dilated"] = dilated_masks
                example["instance_masked_images"] = example["instance_images"] * (example["mask"] < 0.5)

                # Process the caption prompt.
                if self.use_template and self.tokenizer is not None:
                    input_tok = list(self.token_map.values())[0]
                    text = random.choice(self.templates).format(input_tok)
                    example["instance_prompt_ids"] = self.tokenizer(
                        text,
                        padding="do_not_pad",
                        truncation=True,
                        max_length=self.tokenizer.model_max_length,
                    ).input_ids
                # else:
                #     raise NotImplementedError("Only template-based captions are supported.")
                return example
            except Exception as e:
                print("Exception in __getitem__:", e)
                traceback.print_exc()
                continue
