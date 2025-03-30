from abc import ABC, abstractmethod
from typing import Iterator
from utils import logger
from typing import Iterator, TypeVar, Generic, Type, get_args, get_origin, Iterable
import threading
import queue
import os
from typing import TypeVar, Generic

DataFrame = TypeVar("DataFrame")

class DataSource(Generic[DataFrame], Iterator[DataFrame], ABC):
    """
    Abstract class for a data source.
    Provides multi-threaded fetching with a queue-based buffer.
    """
    def __init__(self, buffer_size: int = None, num_workers: int = None) -> None:
        super().__init__()
        
        self.__set_property_types__()
        
        self._buffer_size = buffer_size if buffer_size is not None else 2 * (os.cpu_count() or 2)
        self._num_workers = num_workers if num_workers is not None else (os.cpu_count() or 2)
        
        self.queue = queue.Queue(self._buffer_size)
        self.__should_put_into_queue__ = True
        self._stop_event = threading.Event()
        self._threads = []
        
        # Launch worker threads
        for _ in range(self._num_workers):
            thread = threading.Thread(target=self._worker, daemon=True)
            thread.start()
            self._threads.append(thread)
            
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
                if origin is DataSource:
                    args = get_args(base)
                    if args:
                        self.dataframe_class = args[0]
                        break
        
        print(f"DataSource: Detected DataFrame class: {self.dataframe_class}")

    def _worker(self) -> None:
        """
        Worker thread that repeatedly calls `fetch()` and places items into our queue.
        """
        while not self._stop_event.is_set():
            try:
                data = self.fetch()
                if data is None:
                    # A return of None signals no more data
                    break
                if self.__should_put_into_queue__:
                    # print(f"[DataSource] Worker putting data into queue: {data}")
                    self.queue.put(data)
            except Exception as e:
                # In production, handle/log errors as appropriate
                print(f"[DataSource] Worker error: {e}")
                break
        
        # self._stop_event.set()
        # print("[DataSource] Worker thread stopping.")

    @abstractmethod
    def fetch(self) -> DataFrame:
        """
        Called by each worker thread to get one new item of data.
        Return `None` to signal no more data to fetch.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Stop the data source and free resources.
        """
        pass

    def __next__(self) -> DataFrame:
        """
        Get the next item from our queue in a blocking manner.
        Raises StopIteration if no more items and workers are stopped.
        """
        while True:
            if self._stop_event.is_set() and self.queue.empty():
                raise StopIteration
            try:
                return self.queue.get(timeout=1)
            except queue.Empty:
                # The queue is empty but maybe not truly done
                # Keep looping until the stop event is definitively set and queue is empty
                continue
    
    def has_data(self) -> bool:
        """
        Check if there's data available in the queue.
        Returns True if there's at least one item, False otherwise.
        """
        return not self.queue.empty()

    def __iter__(self) -> Iterator[DataFrame]:
        return self
    def __close__(self) -> None:
        """Helper that sets stop-event, waits for threads to exit, and calls `close()`."""
        self._stop_event.set()
        for t in self._threads:
            t.join()
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__close__()

    @property
    def dataframe_type(self) -> Type[DataFrame]:
        """Return the type of DataFrame this DataSource provides."""
        return self.dataframe_class