"""
Universal dataset converter to FlaxDiff ArrayRecord format.

Converts pre-downloaded datasets (pixparse/cc12m-wds, DiffusionDB, JourneyDB,
CommonCatalog) into the exact ArrayRecord format used by FlaxDiff's training
pipeline and img2dataset.

Format: pack_dict_of_byte_arrays({key: bytes, jpg: bytes, txt: bytes, meta: bytes})

Usage:
    python convert_to_arrayrecord.py --dataset commoncatalog --output_folder gs://bucket/path
    python convert_to_arrayrecord.py --dataset cc12m_wds --output_folder /local/path
    python convert_to_arrayrecord.py --dataset diffusiondb --output_folder gs://bucket/path
    python convert_to_arrayrecord.py --dataset journeydb --output_folder gs://bucket/path
"""

import struct
import os
import json
import io
import argparse
import time
import traceback
from typing import Iterator, Tuple, Dict, Any, Optional
from array_record.python.array_record_module import ArrayRecordWriter
import numpy as np
import cv2
import pyarrow
import pyarrow.fs
import tqdm

# ---- Exact pack/unpack from img2dataset & data-processing.py ----

def pack_dict_of_byte_arrays(data_dict):
    packed_data = bytearray()
    for key, byte_array in data_dict.items():
        if not isinstance(key, str):
            raise ValueError("Keys must be strings")
        key_bytes = key.encode('utf-8')
        packed_data.extend(struct.pack('I', len(key_bytes)))
        packed_data.extend(key_bytes)
        packed_data.extend(struct.pack('I', len(byte_array)))
        packed_data.extend(byte_array)
    return bytes(packed_data)


# ---- ArrayRecord writer (same as img2dataset and data-processing.py) ----

class ArrayRecordSampleWriter:
    def __init__(self, shard_id, output_folder, oom_shard_count=5):
        self.oom_shard_count = oom_shard_count
        shard_name = f"{shard_id:0{oom_shard_count}d}"
        self.output_file = f"{output_folder}/{shard_name}.array_record"
        if "gs:" in output_folder:
            self.tmp_file = f'/tmp/{shard_name}.array_record'
        else:
            self.tmp_file = self.output_file
        self.writer = ArrayRecordWriter(self.tmp_file, options="group_size:1")
        self.count = 0

    def write(self, img_bytes, key_str, caption_str, meta_dict):
        if img_bytes is None:
            return False
        sample = {
            "key": key_str.encode('utf-8') if isinstance(key_str, str) else key_str,
            "jpg": img_bytes if isinstance(img_bytes, bytes) else bytes(img_bytes),
        }
        sample["txt"] = caption_str.encode('utf-8') if isinstance(caption_str, str) else (caption_str or b"")
        for k, v in meta_dict.items():
            if isinstance(v, np.ndarray):
                meta_dict[k] = v.tolist()
        sample["meta"] = json.dumps(meta_dict).encode('utf-8')
        self.writer.write(pack_dict_of_byte_arrays(sample))
        self.count += 1
        return True

    def close(self):
        self.writer.close()
        if self.tmp_file != self.output_file:
            pyarrow.fs.copy_files(self.tmp_file, self.output_file, chunk_size=2**24)
            os.remove(self.tmp_file)


# ---- Image processing (matches img2dataset and data-processing.py exactly) ----

def process_image(img_bytes, target_size=256, encode_quality=95):
    """Decode, resize, re-encode image to JPEG. Returns None on failure."""
    try:
        img_array = np.asarray(bytearray(img_bytes), dtype="uint8")
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]
        if min(h, w) < 100:
            return None
        if max(h, w) / min(h, w) > 2.4:
            return None
        # Resize longest side to target_size, then pad (same as img2dataset border mode)
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        # Pad to target_size x target_size with white border
        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=[255, 255, 255])
        _, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
        return encoded.tobytes()
    except Exception:
        return None


# ---- Dataset Adapters ----

class CommonCatalogAdapter:
    """common-canvas/commoncatalog-cc-by - Parquet with jpg bytes embedded."""

    def __init__(self, split="train", streaming=True):
        from datasets import load_dataset
        self.ds = load_dataset("common-canvas/commoncatalog-cc-by",
                               split=split, streaming=streaming)

    def iterate(self) -> Iterator[Tuple[bytes, str, dict]]:
        for sample in self.ds:
            try:
                # Image is a PIL Image object in HF datasets
                img = sample.get("jpg") or sample.get("image")
                if img is None:
                    continue
                if hasattr(img, 'tobytes'):
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=95)
                    jpg_bytes = buf.getvalue()
                elif isinstance(img, bytes):
                    jpg_bytes = img
                else:
                    continue
                caption = sample.get("caption") or sample.get("blip2_caption") or ""
                meta = {}
                for k in ["key", "url", "width", "height", "sha256", "original_width", "original_height"]:
                    if k in sample and sample[k] is not None:
                        meta[k] = sample[k]
                key = str(sample.get("key") or sample.get("sha256") or hash(caption))
                yield jpg_bytes, str(caption), key, meta
            except Exception:
                continue


class CC12MWDSAdapter:
    """pixparse/cc12m-wds - WebDataset tars with .jpg + .txt + .json per sample."""

    def __init__(self, streaming=True):
        import webdataset as wds
        from datasets import load_dataset
        # Load as HF streaming dataset (handles tar extraction)
        self.ds = load_dataset("pixparse/cc12m-wds", split="train", streaming=streaming)

    def iterate(self) -> Iterator[Tuple[bytes, str, dict]]:
        for sample in self.ds:
            try:
                img = sample.get("jpg") or sample.get("image")
                if img is None:
                    continue
                if hasattr(img, 'tobytes') or hasattr(img, 'save'):
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=95)
                    jpg_bytes = buf.getvalue()
                elif isinstance(img, bytes):
                    jpg_bytes = img
                else:
                    continue
                caption = sample.get("txt", "") or sample.get("caption", "")
                if isinstance(caption, bytes):
                    caption = caption.decode('utf-8', errors='replace')
                json_meta = sample.get("json", {})
                if isinstance(json_meta, str):
                    json_meta = json.loads(json_meta)
                elif isinstance(json_meta, bytes):
                    json_meta = json.loads(json_meta.decode('utf-8'))
                meta = json_meta if isinstance(json_meta, dict) else {}
                key = str(sample.get("__key__") or meta.get("key") or hash(caption))
                yield jpg_bytes, str(caption), key, meta
            except Exception:
                continue


class DiffusionDBAdapter:
    """poloclub/diffusiondb - 2M subset with images and prompts."""

    def __init__(self, subset="2m_random_1k", streaming=True):
        from datasets import load_dataset
        self.ds = load_dataset("poloclub/diffusiondb", subset, split="train",
                               streaming=streaming, trust_remote_code=True)

    def iterate(self) -> Iterator[Tuple[bytes, str, dict]]:
        for sample in self.ds:
            try:
                img = sample.get("image")
                if img is None:
                    continue
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=95)
                jpg_bytes = buf.getvalue()
                caption = sample.get("prompt", "")
                meta = {}
                for k in ["seed", "step", "cfg", "sampler", "width", "height"]:
                    if k in sample and sample[k] is not None:
                        meta[k] = sample[k]
                key = sample.get("image_name", str(hash(caption)))
                yield jpg_bytes, str(caption), str(key), meta
            except Exception:
                continue


class JourneyDBAdapter:
    """JourneyDB/JourneyDB - Gated HF dataset with Midjourney images."""

    def __init__(self, split="train", streaming=True):
        from datasets import load_dataset
        self.ds = load_dataset("JourneyDB/JourneyDB", split=split, streaming=streaming,
                               trust_remote_code=True)

    def iterate(self) -> Iterator[Tuple[bytes, str, dict]]:
        for sample in self.ds:
            try:
                img = sample.get("image")
                if img is None:
                    continue
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=95)
                jpg_bytes = buf.getvalue()
                caption = sample.get("prompt") or sample.get("caption") or ""
                meta = {}
                key = str(hash(str(caption) + str(time.time())))
                yield jpg_bytes, str(caption), key, meta
            except Exception:
                continue


ADAPTERS = {
    "commoncatalog": CommonCatalogAdapter,
    "cc12m_wds": CC12MWDSAdapter,
    "diffusiondb": DiffusionDBAdapter,
    "journeydb": JourneyDBAdapter,
}


# ---- Main conversion logic ----

def convert_dataset(
    adapter,
    output_folder: str,
    num_samples_per_shard: int = 50000,
    image_size: int = 256,
    wandb_project: str = None,
    wandb_entity: str = None,
    dataset_name: str = "unknown",
):
    """Convert a dataset to ArrayRecord format with wandb tracking."""

    # wandb setup
    run = None
    if wandb_project:
        import wandb
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"convert-{dataset_name}",
            config={
                "dataset": dataset_name,
                "output_folder": output_folder,
                "num_samples_per_shard": num_samples_per_shard,
                "image_size": image_size,
            }
        )

    os.makedirs(output_folder, exist_ok=True) if not output_folder.startswith("gs://") else None

    shard_id = 0
    samples_in_shard = 0
    total_written = 0
    total_skipped = 0
    total_processed = 0
    writer = None
    start_time = time.time()

    try:
        pbar = tqdm.tqdm(desc=f"Converting {dataset_name}", unit="samples")
        for jpg_bytes, caption, key, meta in adapter.iterate():
            total_processed += 1

            # Process/resize image
            processed_jpg = process_image(jpg_bytes, target_size=image_size)
            if processed_jpg is None:
                total_skipped += 1
                if total_processed % 10000 == 0:
                    pbar.set_postfix({
                        "written": total_written,
                        "skipped": total_skipped,
                        "shard": shard_id
                    })
                continue

            # Create new shard writer if needed
            if writer is None or samples_in_shard >= num_samples_per_shard:
                if writer is not None:
                    writer.close()
                    shard_id += 1
                    samples_in_shard = 0
                writer = ArrayRecordSampleWriter(shard_id, output_folder)

            # Write sample
            writer.write(processed_jpg, key, caption, meta)
            samples_in_shard += 1
            total_written += 1
            pbar.update(1)

            # Log to wandb periodically
            if run and total_processed % 5000 == 0:
                elapsed = time.time() - start_time
                run.log({
                    "total_processed": total_processed,
                    "total_written": total_written,
                    "total_skipped": total_skipped,
                    "success_rate": total_written / max(total_processed, 1),
                    "samples_per_sec": total_written / max(elapsed, 1),
                    "current_shard": shard_id,
                    "elapsed_minutes": elapsed / 60,
                })

            pbar.set_postfix({
                "written": total_written,
                "skipped": total_skipped,
                "shard": shard_id,
                "rate": f"{total_written / max(time.time() - start_time, 1):.0f}/s"
            })

    except KeyboardInterrupt:
        print(f"\nInterrupted. Saving current shard...")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        if writer is not None:
            writer.close()
        pbar.close()

    elapsed = time.time() - start_time
    summary = {
        "dataset": dataset_name,
        "total_processed": total_processed,
        "total_written": total_written,
        "total_skipped": total_skipped,
        "total_shards": shard_id + 1 if total_written > 0 else 0,
        "success_rate": total_written / max(total_processed, 1),
        "elapsed_minutes": elapsed / 60,
        "samples_per_sec": total_written / max(elapsed, 1),
    }

    print(f"\n=== Conversion Complete ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if run:
        run.summary.update(summary)
        # Log final stats as a wandb table
        import wandb
        table = wandb.Table(columns=list(summary.keys()), data=[list(summary.values())])
        run.log({"conversion_summary": table})
        run.finish()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert datasets to FlaxDiff ArrayRecord format")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(ADAPTERS.keys()),
                        help="Dataset to convert")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output folder (local or gs://bucket/path)")
    parser.add_argument("--num_samples_per_shard", type=int, default=50000,
                        help="Samples per ArrayRecord shard")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Target image size (longest side)")
    parser.add_argument("--wandb_project", type=str, default="msml612-project",
                        help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    args = parser.parse_args()

    adapter_cls = ADAPTERS[args.dataset]
    adapter = adapter_cls()

    convert_dataset(
        adapter=adapter,
        output_folder=args.output_folder,
        num_samples_per_shard=args.num_samples_per_shard,
        image_size=args.image_size,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_entity=args.wandb_entity,
        dataset_name=args.dataset,
    )
