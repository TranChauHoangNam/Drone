import types
import numpy as np

class YOLO:
    """Lightweight stub of ultralytics.YOLO for testing.
    Provides minimal API used in solution.py and tests.
    """
    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.names = {0: "person", 1: "person"}

    def predict(self, img, conf: float = 0.0, verbose: bool = False):
        dummy_box = types.SimpleNamespace(
            cls=np.array([0]),
            conf=np.array([0.9]),
            xyxy=np.array([[0.0, 0.0, 10.0, 10.0]])
        )
        result = types.SimpleNamespace(boxes=[dummy_box])
        return [result]
