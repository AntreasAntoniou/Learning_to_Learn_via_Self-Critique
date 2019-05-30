from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import *
from utils.parser_utils import get_args
from utils.dataset_tools import check_download_dataset

# Combines the arguments, model, data and experiment builders to run an experiment
args, device = get_args()

model = EmbeddingMAMLFewShotClassifier(args=args, device=device,
                                           im_shape=(2, args.image_channels,
                                                     args.image_height, args.image_width))
check_download_dataset(args=args)
data = MetaLearningSystemDataLoader
maml_system = ExperimentBuilder(use_features_instead_of_images=False, model=model, data=data, args=args, device=device)
maml_system.run_experiment()
