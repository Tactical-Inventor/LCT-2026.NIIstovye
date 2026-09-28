"""obstacle_detector: railway obstacle detection from sequential LiDAR clouds."""
from .pipeline import Config
from .stream import StreamProcessor
from .result import StreamResult
from .gauge import GaugeConfig

__version__ = "5.1.0"
__all__ = ["Config", "GaugeConfig", "StreamProcessor", "StreamResult"]
