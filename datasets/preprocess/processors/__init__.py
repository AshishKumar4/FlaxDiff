from abc import ABC, abstractmethod
from typing import Iterator
from utils import logger
from typing import Iterator, TypeVar, Generic, Type, get_args, get_origin, Iterable
import threading
import queue
from sources import DataSource
from typing import TypeVar, Generic, Callable

InputDataFrame = TypeVar("DataInputDataFrameFrame")
OutputDataFrame = TypeVar("DataOutputDataFrame")

class DataProcessor(DataSource[OutputDataFrame], Generic[InputDataFrame, OutputDataFrame], ABC):
    """
    A DataProcessor is also a DataSource.
    It takes one or more upstream data sources and:
      1) Fetches data from them (round-robin or sequentially).
      2) Calls the user-defined `process(...)` on each item.
      3) Emits the processed items as the new DataFrame.
    """

    def __init__(
        self,
        sources: Iterable[DataSource[InputDataFrame]],
        on_success: Callable = None,
        on_error: Callable = None,
        *args,
        **kwargs,
    ):
        self.sources = list(sources)
        self._source_idx = 0
        self.on_success = on_success
        self.on_error = on_error
        super().__init__(*args, **kwargs)

    def __set_property_types__(self) -> None:
        self.dataframe_class = None
        # Try to extract from __orig_class__:
        if hasattr(self, '__orig_class__'):
            args = get_args(self.__orig_class__)
            if args:
                self.dataframe_class = args[0]
        # Fallback to __orig_bases__:
        if self.dataframe_class is None:
            for base in getattr(self.__class__, '__orig_bases__', []):
                origin = get_origin(base)
                if origin is DataProcessor:
                    args = get_args(base)
                    if args:
                        self.dataframe_class = args[-1]
                        break
        # print(f"DataProcessor: Detected DataFrame class: {self.dataframe_class}")
        
    @abstractmethod
    def process(self, data: InputDataFrame, threadId) -> OutputDataFrame:
        """
        User-defined transformation on a single data item.
        Must be overridden by the user.
        """
        pass

    def fetch(self, threadId) -> OutputDataFrame:
        """
        Called by each worker in the parent class.
        We'll pull data from the next available source,
        apply `process`, and return the transformed item.
        Return `None` once all sources are exhausted.
        """
        # Get the first source that has data available
        while self.sources:
            select_source = None
            for source in self.sources:
                if source.has_data():
                    select_source = source
                    break
            if select_source is None:
                # No sources have data available, chose the next one (round-robin)
                select_source = self.sources[self._source_idx % len(self.sources)]
                self._source_idx += 1
            try:
                raw_data = next(select_source)
            except StopIteration:
                # This source is done; remove it
                self.sources.remove(select_source)
                continue
            except Exception as e:
                # Log, skip, or handle accordingly
                print(f"[DataProcessor] Upstream source error: {e}")
                # Optionally remove the source or keep it – depends on your design
                self.sources.remove(select_source)
                continue

            # Transform the raw_data
            processed = self.process(raw_data, threadId=threadId)
            if processed:
                if self.on_success:
                    # Call the success callback if provided
                    self.on_success(processed)
            elif self.on_error:
                # Call the error callback if provided
                self.on_error(raw_data)
            return processed

        # If we get here, all sources are exhausted:
        return None

    def close(self) -> None:
        """
        Close all upstream sources and release resources.
        """
        for src in self.sources:
            src.__close__()  # or just src.close(), if that calls the same logic
        self.sources.clear()
