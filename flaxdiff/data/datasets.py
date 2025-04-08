import jax.numpy as jnp
import grain.python as pygrain
from typing import Dict
import numpy as np
import jax
from flaxdiff.utils import convert_to_global_tree, AutoTextTokenizer
from .dataset_map import datasetMap, onlineDatasetMap
import traceback
from .online_loader import OnlineStreamingDataLoader
import queue
from jax.sharding import Mesh
import threading

def batch_mesh_map(mesh):
    class augmenters(pygrain.MapTransform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def map(self, batch) -> Dict[str, jnp.array]:
            return convert_to_global_tree(mesh, batch)
    return augmenters

def get_dataset_grain(
    data_name="cc12m",
    batch_size=64,
    image_scale=256,
    count=None,
    num_epochs=None,
    method=jax.image.ResizeMethod.LANCZOS3,
    worker_count=32,
    read_thread_count=64,
    read_buffer_size=50,
    worker_buffer_size=20,
    seed=0,
    dataset_source="/mnt/gcs_mount/arrayrecord2/cc12m/",
):
    dataset = datasetMap[data_name]
    data_source = dataset["source"](dataset_source)
    augmenter = dataset["augmenter"](image_scale, method)

    local_batch_size = batch_size // jax.process_count()

    sampler = pygrain.IndexSampler(
        num_records=len(data_source) if count is None else count,
        shuffle=True,
        seed=seed,
        num_epochs=num_epochs,
        shard_options=pygrain.ShardByJaxProcess(),
    )

    def get_trainset():
        transformations = [
            augmenter(),
            pygrain.Batch(local_batch_size, drop_remainder=True),
        ]
        
        # if mesh != None:
        #     transformations += [batch_mesh_map(mesh)]

        loader = pygrain.DataLoader(
            data_source=data_source,
            sampler=sampler,
            operations=transformations,
            worker_count=worker_count,
            read_options=pygrain.ReadOptions(
                read_thread_count, read_buffer_size
            ),
            worker_buffer_size=worker_buffer_size,
        )
        return loader
    

    return {
        "train": get_trainset,
        "train_len": len(data_source),
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
        # "null_labels": null_labels,
        # "null_labels_full": null_labels_full,
        # "model": model,
        # "tokenizer": tokenizer,
    }

def generate_collate_fn():
    auto_tokenize = AutoTextTokenizer(tensor_type="np")
    def default_collate(batch):
        try:
            # urls = [sample["url"] for sample in batch]
            captions = [sample["caption"] for sample in batch]
            results = auto_tokenize(captions)
            images = np.stack([sample["image"] for sample in batch], axis=0)
            return {
                "image": images,
                "input_ids": results['input_ids'],
                "attention_mask": results['attention_mask'],
            }
        except Exception as e:
            print("Error in collate function", e, [sample["image"].shape for sample in batch])
            traceback.print_exc()
            
    return default_collate
    
def get_dataset_online(
        data_name="combined_online",
        batch_size=64,
        image_scale=256,
        count=None,
        num_epochs=None,
        method=jax.image.ResizeMethod.LANCZOS3,
        worker_count=32,
        read_thread_count=64,
        read_buffer_size=50,
        worker_buffer_size=20,
        seed=0,
        dataset_source="/mnt/gcs_mount/arrayrecord2/cc12m/",
    ):
    local_batch_size = batch_size // jax.process_count()
    
    sources = onlineDatasetMap[data_name]["source"]
    dataloader = OnlineStreamingDataLoader(
            sources, 
            batch_size=local_batch_size,
            num_workers=worker_count,
            num_threads=read_thread_count,
            image_shape=(image_scale, image_scale),
            global_process_count=jax.process_count(),
            global_process_index=jax.process_index(),
            prefetch=worker_buffer_size,
            collate_fn=generate_collate_fn(),
            default_split="train",
        )
    
    def get_trainset(mesh: Mesh = None):
        if mesh != None:
            class dataLoaderWithMesh:
                def __init__(self, dataloader, mesh):
                    self.dataloader = dataloader
                    self.mesh = mesh
                    self.tmp_queue = queue.Queue(worker_buffer_size)
                    def batch_loader():
                        for batch in self.dataloader:
                            try:
                                self.tmp_queue.put(convert_to_global_tree(mesh, batch))
                            except Exception as e:
                                print("Error processing batch", e)
                    self.loader_thread = threading.Thread(target=batch_loader)
                    self.loader_thread.start()
                    
                def __iter__(self):
                    return self
                
                def __next__(self):
                    return self.tmp_queue.get()
                
            dataloader_with_mesh = dataLoaderWithMesh(dataloader, mesh)
                    
            return dataloader_with_mesh
        return dataloader
    
    return {
        "train": get_trainset,
        "train_len": len(dataloader) * jax.process_count(),
        "local_batch_size": local_batch_size,
        "global_batch_size": batch_size,
        # "null_labels": null_labels,
        # "null_labels_full": null_labels_full,
        # "model": model,
        # "tokenizer": tokenizer,
    }