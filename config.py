import pathlib
import torch


# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directory paths
ROOT_DIR = pathlib.Path('.')
QUESTIONS_DIR = ROOT_DIR / 'questions'
TOKENIZED_QUESTION_DIR = ROOT_DIR / 'tokenized_questions'
VISUAL_FEATURES_DIR = ROOT_DIR / 'visual_features'
SCENE_GRAPHS_DIR = ROOT_DIR / 'scene_graphs'
META_INFO_DIR = ROOT_DIR / 'meta_info'