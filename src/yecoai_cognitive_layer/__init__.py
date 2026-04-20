from .feature_engine import FeatureEngine
from .model import CognitiveModel
import importlib.resources as pkg_resources

__version__ = "1.0.0"
__all__ = ["FeatureEngine", "CognitiveModel", "get_default_model"]

def get_default_model():
    with pkg_resources.path('yecoai_cognitive_layer', 'weights.json') as weights_path:
        return CognitiveModel.load_from_json(str(weights_path))
