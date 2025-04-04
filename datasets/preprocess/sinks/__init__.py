from abc import ABC, abstractmethod
from typing import Iterator
from utils import logger
from typing import Iterator, TypeVar, Generic, Type, get_args, get_origin, Iterable
import threading
import queue
from sources import DataSource
from processors import DataProcessor
from abc import ABC, abstractmethod
from typing import Iterable, TypeVar, Generic
from sources import DataSource


InputDataFrame = TypeVar("DataInputDataFrameFrame")

# DataSink is generic over DataFrame, just like DataProcessor
class DataSink(DataProcessor[InputDataFrame, None], ABC):
    """
    A DataSink is the final consumer in a data processing pipeline.
    It inherits the multi-threaded, queued processing from DataProcessor,
    but instead of a transformation that returns a new DataFrame, it calls
    a user-defined sink function to consume each data item.
    
    Optionally, the sink() function may return a confirmation or status which
    will be forwarded through the queue.
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__should_put_into_queue__ = False
    
    @abstractmethod
    def write(self, data: InputDataFrame, threadId) -> None:
        """
        User-defined function to consume a data item.
        For example, writing the data to a file, database, or any other endpoint.
        """
        pass

    def process(self, data: InputDataFrame, threadId) -> None:
        """
        Instead of performing a transformation and returning a new item,
        this method calls the sink() function with the data.
        You may optionally return the original data (or a status) if you want
        to pass something downstream; otherwise, the sink side effect is enough.
        """
        self.write(data, threadId=threadId)
        return None