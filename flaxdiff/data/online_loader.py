import multiprocessing
import threading
from multiprocessing import Queue
# from arrayqueues.shared_arrays import ArrayQueue
# from faster_fifo import Queue
import time
import albumentations as A
import queue
import cv2
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
from functools import partial

from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from datasets.utils.file_utils import get_datasets_user_agent
from concurrent.futures import ThreadPoolExecutor
import io
import urllib

import PIL.Image
import cv2
import traceback 

USER_AGENT = get_datasets_user_agent()

data_queue = Queue(16*2000)


def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def default_image_processor(
    image, image_shape, 
    min_image_shape=(128, 128),
    upscale_interpolation=cv2.INTER_CUBIC,
    downscale_interpolation=cv2.INTER_AREA,
):
    image = np.array(image)
    original_height, original_width = image.shape[:2]
    # check if the image is too small
    if min(original_height, original_width) < min(min_image_shape):
        return None, original_height, original_width
    # check if wrong aspect ratio
    if max(original_height, original_width) / min(original_height, original_width) > 2.4:
        return None, original_height, original_width
    # check if the variance is too low
    if np.std(image) < 1e-5:
        return None, original_height, original_width
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    downscale = max(original_width, original_height) > max(image_shape)
    interpolation = downscale_interpolation if downscale else upscale_interpolation

    image = A.longest_max_size(image, max(
        image_shape), interpolation=interpolation)
    image = A.pad(
        image,
        min_height=image_shape[0],
        min_width=image_shape[1],
        border_mode=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    return image, original_height, original_width


def map_sample(
    url, caption,
    image_shape=(256, 256),
    min_image_shape=(128, 128),
    timeout=15,
    retries=3,
    upscale_interpolation=cv2.INTER_CUBIC,
    downscale_interpolation=cv2.INTER_AREA,
    image_processor=default_image_processor,
):
    try:
        # Assuming fetch_single_image is defined elsewhere
        image = fetch_single_image(url, timeout=timeout, retries=retries)
        if image is None:
            return

        image, original_height, original_width = image_processor(
            image, image_shape, min_image_shape=min_image_shape,
            upscale_interpolation=upscale_interpolation,
            downscale_interpolation=downscale_interpolation,)
        
        if image is None:
            return
        
        data_queue.put({
            "url": url,
            "caption": caption,
            "image": image,
            "original_height": original_height,
            "original_width": original_width,
        })
    except Exception as e:
        print(f"Error maping sample {url}", e)
        traceback.print_exc() 
        # error_queue.put_nowait({
        #     "url": url,
        #     "caption": caption,
        #     "error": str(e)
        # })
        pass


def map_batch(
    batch, num_threads=256, image_shape=(256, 256), 
    min_image_shape=(128, 128),
    timeout=15, retries=3, image_processor=default_image_processor,
    upscale_interpolation=cv2.INTER_CUBIC,
    downscale_interpolation=cv2.INTER_AREA,
):
    try:
        map_sample_fn = partial(map_sample, image_shape=image_shape, min_image_shape=min_image_shape,
                                timeout=timeout, retries=retries, image_processor=image_processor,
                                upscale_interpolation=upscale_interpolation,
                                downscale_interpolation=downscale_interpolation)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(map_sample_fn, batch["url"], batch['caption'])
    except Exception as e:
        print(f"Error maping batch", e)
        traceback.print_exc() 
        # error_queue.put_nowait({
        #     "batch": batch,
        #     "error": str(e)
        # })
        pass


def parallel_image_loader(
    dataset: Dataset, num_workers: int = 8, image_shape=(256, 256), 
    min_image_shape=(128, 128),
    num_threads=256, timeout=15, retries=3, image_processor=default_image_processor,
    upscale_interpolation=cv2.INTER_CUBIC,
    downscale_interpolation=cv2.INTER_AREA,
):
    map_batch_fn = partial(map_batch, num_threads=num_threads, image_shape=image_shape, 
                           min_image_shape=min_image_shape,
                           timeout=timeout, retries=retries, image_processor=image_processor,
                           upscale_interpolation=upscale_interpolation,
                           downscale_interpolation=downscale_interpolation)
    shard_len = len(dataset) // num_workers
    print(f"Local Shard lengths: {shard_len}")
    with multiprocessing.Pool(num_workers) as pool:
        iteration = 0
        while True:
            # Repeat forever
            shards = [dataset[i*shard_len:(i+1)*shard_len]
                      for i in range(num_workers)]
            print(f"mapping {len(shards)} shards")
            pool.map(map_batch_fn, shards)
            iteration += 1
            print(f"Shuffling dataset with seed {iteration}")
            dataset = dataset.shuffle(seed=iteration)
            # Clear the error queue
            # while not error_queue.empty():
            #     error_queue.get_nowait()


class ImageBatchIterator:
    def __init__(
        self, dataset: Dataset, batch_size: int = 64, image_shape=(256, 256), 
        min_image_shape=(128, 128),
        num_workers: int = 8, num_threads=256, timeout=15, retries=3, 
        image_processor=default_image_processor,
        upscale_interpolation=cv2.INTER_CUBIC,
        downscale_interpolation=cv2.INTER_AREA,
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        loader = partial(parallel_image_loader, num_threads=num_threads,
                         image_shape=image_shape,
                         min_image_shape=min_image_shape, 
                         num_workers=num_workers, 
                         timeout=timeout, retries=retries, image_processor=image_processor,
                         upscale_interpolation=upscale_interpolation,
                         downscale_interpolation=downscale_interpolation)
        self.thread = threading.Thread(target=loader, args=(dataset,))
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        def fetcher(_):
            return data_queue.get()
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            batch = list(executor.map(fetcher, range(self.batch_size)))
        return batch

    def __del__(self):
        self.thread.join()

    def __len__(self):
        return len(self.dataset) // self.batch_size


def default_collate(batch):
    urls = [sample["url"] for sample in batch]
    captions = [sample["caption"] for sample in batch]
    images = np.stack([sample["image"] for sample in batch], axis=0)
    return {
        "url": urls,
        "caption": captions,
        "image": images,
    }


def dataMapper(map: Dict[str, Any]):
    def _map(sample) -> Dict[str, Any]:
        return {
            "url": sample[map["url"]],
            "caption": sample[map["caption"]],
        }
    return _map


class OnlineStreamingDataLoader():
    def __init__(
        self,
        dataset,
        batch_size=64,
        image_shape=(256, 256),
        min_image_shape=(128, 128),
        num_workers=16,
        num_threads=512,
        default_split="all",
        pre_map_maker=dataMapper,
        pre_map_def={
            "url": "URL",
            "caption": "TEXT",
        },
        global_process_count=1,
        global_process_index=0,
        prefetch=1000,
        collate_fn=default_collate,
        timeout=15,
        retries=3,
        image_processor=default_image_processor,
        upscale_interpolation=cv2.INTER_CUBIC,
        downscale_interpolation=cv2.INTER_AREA,
    ):
        if isinstance(dataset, str):
            dataset_path = dataset
            print("Loading dataset from path")
            if "gs://" in dataset:
                dataset = load_from_disk(dataset_path)
            else:
                dataset = load_dataset(dataset_path, split=default_split)
        elif isinstance(dataset, list):
            if isinstance(dataset[0], str):
                print("Loading multiple datasets from paths")
                dataset = [load_from_disk(dataset_path) if "gs://" in dataset_path else load_dataset(
                    dataset_path, split=default_split) for dataset_path in dataset]
            print("Concatenating multiple datasets")
            dataset = concatenate_datasets(dataset)
            dataset = dataset.shuffle(seed=0)
        # dataset = dataset.map(pre_map_maker(pre_map_def), batched=True, batch_size=10000000)
        self.dataset = dataset.shard(
            num_shards=global_process_count, index=global_process_index)
        print(f"Dataset length: {len(dataset)}")
        self.iterator = ImageBatchIterator(self.dataset, image_shape=image_shape,
                                           min_image_shape=min_image_shape,
                                           num_workers=num_workers, batch_size=batch_size, num_threads=num_threads,
                                            timeout=timeout, retries=retries, image_processor=image_processor,
                                             upscale_interpolation=upscale_interpolation,
                                             downscale_interpolation=downscale_interpolation)
        self.batch_size = batch_size

        # Launch a thread to load batches in the background
        self.batch_queue = queue.Queue(prefetch)

        def batch_loader():
            for batch in self.iterator:
                try:
                    self.batch_queue.put(collate_fn(batch))
                except Exception as e:
                    print("Error processing batch", e)

        self.loader_thread = threading.Thread(target=batch_loader)
        self.loader_thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        return self.batch_queue.get()
        # return self.collate_fn(next(self.iterator))

    def __len__(self):
        return len(self.dataset)