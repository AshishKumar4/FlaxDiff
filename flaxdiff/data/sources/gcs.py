import cv2
import jax.numpy as jnp
import grain.python as pygrain
from flaxdiff.utils import AutoTextTokenizer
from typing import Dict
import os
import struct as st
from functools import partial
import numpy as np

# -----------------------------------------------------------------------------------------------#
# CC12m and other GCS data sources --------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------#

def data_source_gcs(source='arrayrecord/laion-aesthetics-12m+mscoco-2017'):
    def data_source(base="/home/mrwhite0racle/gcs_mount"):
        records_path = os.path.join(base, source)
        records = [os.path.join(records_path, i) for i in os.listdir(
            records_path) if 'array_record' in i]
        ds = pygrain.ArrayRecordDataSource(records)
        return ds
    return data_source

def data_source_combined_gcs(
    sources=[]):
    def data_source(base="/home/mrwhite0racle/gcs_mount"):
        records_paths = [os.path.join(base, source) for source in sources]
        records = []
        for records_path in records_paths:
            records += [os.path.join(records_path, i) for i in os.listdir(
                records_path) if 'array_record' in i]
        ds = pygrain.ArrayRecordDataSource(records)
        return ds
    return data_source

def unpack_dict_of_byte_arrays(packed_data):
    unpacked_dict = {}
    offset = 0
    while offset < len(packed_data):
        # Unpack the key length
        key_length = st.unpack_from('I', packed_data, offset)[0]
        offset += st.calcsize('I')
        # Unpack the key bytes and convert to string
        key = packed_data[offset:offset+key_length].decode('utf-8')
        offset += key_length
        # Unpack the byte array length
        byte_array_length = st.unpack_from('I', packed_data, offset)[0]
        offset += st.calcsize('I')
        # Unpack the byte array
        byte_array = packed_data[offset:offset+byte_array_length]
        offset += byte_array_length
        unpacked_dict[key] = byte_array
    return unpacked_dict

def image_augmenter(image, image_scale, method=cv2.INTER_AREA):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_scale, image_scale),
                            interpolation=cv2.INTER_AREA)
    return image

def gcs_augmenters(image_scale, method):
    labelizer = lambda sample : sample['txt']
    class augmenters(pygrain.MapTransform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.auto_tokenize = AutoTextTokenizer(tensor_type="np")
            self.image_augmenter = partial(image_augmenter, image_scale=image_scale, method=method)

        def map(self, element) -> Dict[str, jnp.array]:
            element = unpack_dict_of_byte_arrays(element)
            image = np.asarray(bytearray(element['jpg']), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            image = self.image_augmenter(image)
            caption = labelizer(element).decode('utf-8')
            results = self.auto_tokenize(caption)
            return {
                "image": image,
                "input_ids": results['input_ids'][0],
                "attention_mask": results['attention_mask'][0],
            }
    return augmenters
