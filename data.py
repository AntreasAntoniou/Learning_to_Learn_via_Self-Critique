import concurrent.futures
import json
import os
import pickle

import numpy as np
import torch
import tqdm
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def augment_image(image):
    transforms_to_use = [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    cur_image = np.copy(image)
    for transform_current in transforms_to_use:
        cur_image = transform_current(cur_image)

    return cur_image


class FewShotLearningDatasetParallel(Dataset):
    def __init__(self, args, model=None):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param args: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.data_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.image_height, self.image_width, self.image_channel = args.image_height, args.image_width, args.image_channels
        self.args = args
        self.indexes_of_folders_indicating_class = args.indexes_of_folders_indicating_class
        self.reverse_channels = args.reverse_channels
        self.labels_as_int = args.labels_as_int
        self.train_val_test_split = args.train_val_test_split
        self.current_set_name = "train"
        self.num_target_samples = args.num_target_samples
        self.reset_stored_filepaths = args.reset_stored_filepaths
        val_rng = np.random.RandomState(seed=args.val_seed)
        self.model = model
        val_seed = val_rng.randint(1, 999999)
        train_rng = np.random.RandomState(seed=args.train_seed)
        train_seed = train_rng.randint(1, 999999)
        test_rng = np.random.RandomState(seed=args.val_seed)
        test_seed = test_rng.randint(1, 999999)
        args.val_seed = val_seed
        args.train_seed = train_seed
        args.test_seed = test_seed
        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}
        self.seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.val_seed}
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.augment_images = False
        self.num_samples_per_class = args.num_samples_per_class
        self.num_classes_per_set = args.num_classes_per_set

        self.rng = np.random.RandomState(seed=self.seed['val'])
        self.datasets = self.load_dataset()

        self.indexes = {"train": 0, "val": 0, 'test': 0}
        self.dataset_size_dict = {
            set_name: {key: len(self.datasets[set_name][key]) for key in list(self.datasets[set_name].keys())} for
            set_name in self.datasets.keys()}

        self.label_set = self.get_label_set()
        self.data_length = {name: np.sum([len(self.datasets[name][key])
                                          for key in self.datasets[name]]) for name in self.datasets.keys()}

        print("data", self.data_length)
        self.observed_seed_set = None

    def load_dataset(self):
        """
        Loads a dataset's dictionary files and splits the data according to the train_val_test_split variable stored
        in the args object.
        :return: Three sets, the training set, validation set and test sets (referred to as the meta-train,
        meta-val and meta-test in the paper)
        """
        rng = np.random.RandomState(seed=self.seed['val'])

        if self.args.sets_are_pre_split == True:
            data_image_paths, index_to_label_name_dict_file, label_to_index = self.load_datapaths()
            dataset_splits = dict()
            for key, value in data_image_paths.items():
                key = self.get_label_from_index(index=key)
                bits = key.split("/")
                set_name = bits[0]
                class_label = bits[1]
                if set_name not in dataset_splits:
                    dataset_splits[set_name] = {class_label: value}
                else:
                    dataset_splits[set_name][class_label] = value

        elif self.args.load_from_npz_files == True:
            with open("{}/train_{}.pkl".format(self.args.dataset_path,
                                               self.args.dataset_name.replace("_pkl", "").replace("_", "-")),
                      mode="rb") as x_train_data:
                x_train = pickle.load(x_train_data)

            with open("{}/val_{}.pkl".format(self.args.dataset_path,
                                             self.args.dataset_name.replace("_pkl", "").replace("_", "-")),
                      mode="rb") as x_val_data:
                x_val = pickle.load(x_val_data)

            with open("{}/test_{}.pkl".format(self.args.dataset_path,
                                              self.args.dataset_name.replace("_pkl", "").replace("_", "-")),
                      mode="rb") as x_test_data:
                x_test = pickle.load(x_test_data)

            dataset_splits = {"train": x_train, "val": x_val, "test": x_test}
            labels = sorted(list(x_train.keys()) + list(x_val.keys()) + list(x_test.keys()))
            idx_to_label_name = {idx: label for idx, label in enumerate(labels)}
            label_name_to_idx = {label: idx for idx, label in enumerate(labels)}
            self.index_to_label_name_dict_file = "{}/map_to_label_name_{}.json".format(self.args.dataset_path,
                                                                                       self.dataset_name)
            self.label_name_to_map_dict_file = "{}/label_name_to_map_{}.json".format(self.args.dataset_path,
                                                                                     self.dataset_name)
            self.save_to_json(dict_to_store=idx_to_label_name, filename=self.index_to_label_name_dict_file)
            self.save_to_json(dict_to_store=label_name_to_idx, filename=self.label_name_to_map_dict_file)
            print(x_train.keys(), x_val.keys())

        else:
            data_image_paths, index_to_label_name_dict_file, label_to_index = self.load_datapaths()
            total_label_types = len(data_image_paths)
            num_classes_idx = np.arange(len(data_image_paths.keys()), dtype=np.int32)
            rng.shuffle(num_classes_idx)
            keys = list(data_image_paths.keys())
            values = list(data_image_paths.values())
            new_keys = [keys[idx] for idx in num_classes_idx]
            new_values = [values[idx] for idx in num_classes_idx]
            data_image_paths = dict(zip(new_keys, new_values))
            # data_image_paths = self.shuffle(data_image_paths)
            x_train_id, x_val_id, x_test_id = int(self.train_val_test_split[0] * total_label_types), \
                                              int(np.sum(self.train_val_test_split[:2]) * total_label_types), \
                                              int(total_label_types)
            print(x_train_id, x_val_id, x_test_id)
            x_train_classes = (class_key for class_key in list(data_image_paths.keys())[:x_train_id])
            x_val_classes = (class_key for class_key in list(data_image_paths.keys())[x_train_id:x_val_id])
            x_test_classes = (class_key for class_key in list(data_image_paths.keys())[x_val_id:x_test_id])
            x_train, x_val, x_test = {class_key: data_image_paths[class_key] for class_key in x_train_classes}, \
                                     {class_key: data_image_paths[class_key] for class_key in x_val_classes}, \
                                     {class_key: data_image_paths[class_key] for class_key in x_test_classes},
            dataset_splits = {"train": x_train, "val": x_val, "test": x_test}

        if self.args.load_into_memory is True:
            print("Loading data into RAM")
            x_loaded = {"train": [], "val": [], "test": []}

            for set_key, set_value in dataset_splits.items():
                print("Currently loading into memory the {} set".format(set_key))
                x_loaded[set_key] = {key: np.zeros(len(value), ) for key, value in set_value.items()}
                # for class_key, class_value in set_value.items():
                with tqdm.tqdm(total=len(set_value)) as pbar_memory_load:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                        # Process the list of files, but split the work across the process pool to use all CPUs!
                        for (class_label, class_images_loaded) in executor.map(self.load_parallel_batch,
                                                                               (set_value.items())):
                            x_loaded[set_key][class_label] = class_images_loaded
                            pbar_memory_load.update(1)

            dataset_splits = x_loaded
            self.args.load_from_npz_files = True


        return dataset_splits

    def load_datapaths(self):
        """
        If saved json dictionaries of the data are available, then this method loads the dictionaries such that the
        data is ready to be read. If the json dictionaries do not exist, then this method calls get_data_paths()
        which will build the json dictionary containing the class to filepath samples, and then store them.
        :return: data_image_paths: dict containing class to filepath list pairs.
                 index_to_label_name_dict_file: dict containing numerical indexes mapped to the human understandable
                 string-names of the class
                 label_to_index: dictionary containing human understandable string mapped to numerical indexes
        """
        dataset_dir = os.environ['DATASET_DIR']
        data_path_file = "{}/{}.json".format(dataset_dir, self.dataset_name)
        self.index_to_label_name_dict_file = "{}/map_to_label_name_{}.json".format(dataset_dir, self.dataset_name)
        self.label_name_to_map_dict_file = "{}/label_name_to_map_{}.json".format(dataset_dir, self.dataset_name)

        if not os.path.exists(data_path_file):
            self.reset_stored_filepaths = True

        if self.reset_stored_filepaths == True:
            if os.path.exists(data_path_file):
                os.remove(data_path_file)
            self.reset_stored_filepaths = False

        try:
            data_image_paths = self.load_from_json(filename=data_path_file)
            label_to_index = self.load_from_json(filename=self.label_name_to_map_dict_file)
            index_to_label_name_dict_file = self.load_from_json(filename=self.index_to_label_name_dict_file)
            return data_image_paths, index_to_label_name_dict_file, label_to_index
        except:
            print("Mapped data paths can't be found, remapping paths..")
            data_image_paths, code_to_label_name, label_name_to_code = self.get_data_paths()
            self.save_to_json(dict_to_store=data_image_paths, filename=data_path_file)
            self.save_to_json(dict_to_store=code_to_label_name, filename=self.index_to_label_name_dict_file)
            self.save_to_json(dict_to_store=label_name_to_code, filename=self.label_name_to_map_dict_file)
            return self.load_datapaths()

    def save_to_json(self, filename, dict_to_store):
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(dict_to_store, fp=f)

    def load_from_json(self, filename):
        with open(filename, mode="r") as f:
            load_dict = json.load(fp=f)

        return load_dict

    def load_test_image(self, filepath):
        """
        Tests whether a target filepath contains an uncorrupted image. If image is corrupted, attempt to fix.
        :param filepath: Filepath of image to be tested
        :return: Return filepath of image if image exists and is uncorrupted (or attempt to fix has succeeded),
        else return None
        """
        image = None
        try:
            image = Image.open(filepath)
        except RuntimeWarning:
            os.system("convert {} -strip {}".format(filepath, filepath))
            print("converting")
            image = Image.open(filepath)
        except:
            print("Broken image")

        if image is not None:
            return filepath
        else:
            return None

    def get_data_paths(self):
        """
        Method that scans the dataset directory and generates class to image-filepath list dictionaries.
        :return: data_image_paths: dict containing class to filepath list pairs.
                 index_to_label_name_dict_file: dict containing numerical indexes mapped to the human understandable
                 string-names of the class
                 label_to_index: dictionary containing human understandable string mapped to numerical indexes
        """
        print("Get images from", self.data_path)
        data_image_path_list_raw = []
        labels = set()
        for subdir, dir, files in os.walk(self.data_path):
            for file in files:
                if (".jpeg") in file.lower() or (".png") in file.lower() or (".jpg") in file.lower():
                    filepath = os.path.join(subdir, file)
                    label = self.get_label_from_path(os.path.abspath(filepath))
                    data_image_path_list_raw.append(filepath)
                    labels.add(label)

        labels = sorted(labels)
        idx_to_label_name = {idx: label for idx, label in enumerate(labels)}
        label_name_to_idx = {label: idx for idx, label in enumerate(labels)}
        data_image_path_dict = {idx: [] for idx in list(idx_to_label_name.keys())}
        with tqdm.tqdm(total=len(data_image_path_list_raw)) as pbar_error:
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                # Process the list of files, but split the work across the process pool to use all CPUs!
                for image_file in executor.map(self.load_test_image, (data_image_path_list_raw)):
                    pbar_error.update(1)
                    if image_file is not None:
                        label = self.get_label_from_path(image_file)
                        data_image_path_dict[label_name_to_idx[label]].append(image_file)

        return data_image_path_dict, idx_to_label_name, label_name_to_idx

    def get_label_set(self):
        """
        Generates a set containing all class numerical indexes
        :return: A set containing all class numerical indexes
        """
        index_to_label_name_dict_file = self.load_from_json(filename=self.index_to_label_name_dict_file)
        return set(list(index_to_label_name_dict_file.keys()))

    def get_index_from_label(self, label):
        """
        Given a class's (human understandable) string, returns the numerical index of that class
        :param label: A string of a human understandable class contained in the dataset
        :return: An int containing the numerical index of the given class-string
        """
        label_to_index = self.load_from_json(filename=self.label_name_to_map_dict_file)
        return label_to_index[label]

    def get_label_from_index(self, index):
        """
        Given an index return the human understandable label mapping to it.
        :param index: A numerical index (int)
        :return: A human understandable label (str)
        """
        index_to_label_name = self.load_from_json(filename=self.index_to_label_name_dict_file)
        return index_to_label_name[index]

    def get_label_from_path(self, filepath):
        """
        Given a path of an image generate the human understandable label for that image.
        :param filepath: The image's filepath
        :return: A human understandable label.
        """
        label_bits = filepath.split("/")
        label = "/".join([label_bits[idx] for idx in self.indexes_of_folders_indicating_class])
        if self.labels_as_int:
            label = int(label)
        return label

    def load_image(self, image_path, channels):
        """
        Given an image filepath and the number of channels to keep, load an image and keep the specified channels
        :param image_path: The image's filepath
        :param channels: The number of channels to keep
        :return: An image array of shape (h, w, channels), whose values range between 0.0 and 1.0.
        """
        if not self.args.load_from_npz_files:
            image = Image.open(image_path)
            if 'omniglot' in self.dataset_name:
                image = image.resize((self.image_height, self.image_width), resample=Image.LANCZOS)
                image = np.array(image, np.float32)
                if channels == 1:
                    image = np.expand_dims(image, axis=2)
            else:
                image = image.resize((self.image_height, self.image_width)).convert('RGB')
                image = np.array(image, np.float32)
                image = image / 255.0
        else:
            image = image_path

        return image

    def load_batch(self, batch_image_paths):
        """
        Load a batch of images, given a list of filepaths
        :param batch_image_paths: A list of filepaths
        :return: A numpy array of images of shape batch, height, width, channels
        """
        image_batch = []

        if self.args.load_from_npz_files:
            for image_path in batch_image_paths:
                image_batch.append(image_path)
            image_batch = np.array(image_batch, dtype=np.float32)
            # print(image_batch.shape)
        else:
            image_batch = [self.load_image(image_path=image_path, channels=self.image_channel)
                           for image_path in batch_image_paths]
            image_batch = np.array(image_batch, dtype=np.float32)

        return image_batch

    def load_parallel_batch(self, inputs):
        """
        Load a batch of images, given a list of filepaths
        :param batch_image_paths: A list of filepaths
        :return: A numpy array of images of shape batch, height, width, channels
        """
        class_label, batch_image_paths = inputs
        image_batch = []

        if self.args.load_from_npz_files:
            for image_path in batch_image_paths:
                image_batch.append(np.copy(image_path))
            image_batch = np.array(image_batch, dtype=np.float32)
        else:
            # with tqdm.tqdm(total=1) as load_pbar:
            image_batch = [self.load_image(image_path=image_path, channels=self.image_channel)
                           for image_path in batch_image_paths]
            # load_pbar.update(1)

            image_batch = np.array(image_batch, dtype=np.float32)

        return class_label, image_batch

    def reconstruct_original(self, x):
        """
        Applies the reverse operations that preprocess_data() applies such that the data returns to their original form
        :param x: A batch of data to reconstruct
        :return: A reconstructed batch of data
        """
        x = x * 255.0
        return x

    def shuffle(self, x, rng):
        """
        Shuffles the data batch along it's first axis
        :param x: A data batch
        :return: A shuffled data batch
        """
        indices = np.arange(len(x))
        rng.shuffle(indices)
        x = x[indices]
        return x

    def get_set(self, dataset_name, seed, augment_images=False):
        """
        Generates a task-set to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc.
        :return: A task-set containing an image and label support set, and an image and label target set.
        """
        # seed = seed % self.args.total_unique_tasks
        rng = np.random.RandomState(seed)
        selected_classes = rng.choice(list(self.dataset_size_dict[dataset_name].keys()),
                                      size=self.num_classes_per_set, replace=False)
        rng.shuffle(selected_classes)
        # idx = rng.randint(0, 1000000)
        k_list = rng.randint(0, 4, size=self.num_classes_per_set)
        k_dict = {selected_class: k_item for (selected_class, k_item) in zip(selected_classes, k_list)}
        episode_labels = [i for i in range(self.num_classes_per_set)]
        class_to_episode_label = {selected_class: episode_label for (selected_class, episode_label) in
                                  zip(selected_classes, episode_labels)}

        class_label_to_class_idx = {class_label: class_idx for class_idx, class_label in
                                    enumerate(list(self.dataset_size_dict[dataset_name].keys()))}

        support_set_images = []
        support_set_labels = []

        target_set_images = []
        target_set_labels = []

        support_set_original_labels = []
        target_set_original_labels = []

        for class_idx, class_entry in enumerate(selected_classes):
            choose_samples_list = rng.choice(self.dataset_size_dict[dataset_name][class_entry],
                                             size=self.num_samples_per_class + self.num_target_samples, replace=False)
            x_current_class = []
            y_current_class = []

            y_original_class = []
            for sample in choose_samples_list:
                choose_samples = self.datasets[dataset_name][class_entry][sample]

                k = k_dict[class_entry]
                x_class_data = self.load_batch([choose_samples])[0]
                x = augment_image(image=x_class_data)
                y = int(class_to_episode_label[class_entry])
                x_current_class.append(x)
                y_current_class.append(y)
                y_original_class.append(class_label_to_class_idx[class_entry])

            support_set_images.extend(x_current_class[:self.num_samples_per_class])
            support_set_labels.extend(y_current_class[:self.num_samples_per_class])
            support_set_original_labels.extend(y_original_class[:self.num_samples_per_class])

            target_set_images.extend(x_current_class[self.num_samples_per_class:])
            target_set_labels.extend(y_current_class[self.num_samples_per_class:])
            target_set_original_labels.extend(y_original_class[self.num_samples_per_class:])

        support_set_images = torch.stack(support_set_images)
        support_set_labels = torch.Tensor(support_set_labels)
        support_set_original_labels = torch.Tensor(support_set_original_labels)

        target_set_images = torch.stack(target_set_images)
        target_set_labels = torch.Tensor(target_set_labels)
        target_set_original_labels = torch.Tensor(target_set_original_labels)

        support_set_shuffle_ids = rng.shuffle(np.arange(len(support_set_images)))
        support_set_images = support_set_images[support_set_shuffle_ids]
        support_set_labels = support_set_labels[support_set_shuffle_ids]
        support_set_original_labels = support_set_original_labels[support_set_shuffle_ids]

        target_set_shuffle_ids = rng.shuffle(np.arange(len(target_set_images)))
        target_set_images = target_set_images[target_set_shuffle_ids]
        target_set_labels = target_set_labels[target_set_shuffle_ids]
        target_set_original_labels = target_set_original_labels[target_set_shuffle_ids]

        support_set_augmented_images = support_set_images.clone()
        target_set_augmented_images = target_set_images.clone()

        return support_set_images, support_set_augmented_images, target_set_images, target_set_augmented_images, support_set_labels, target_set_labels, \
               support_set_original_labels, target_set_original_labels, seed

    def __len__(self):
        total_samples = self.data_length[self.current_set_name]
        return total_samples

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def set_augmentation(self, augment_images):
        self.augment_images = augment_images

    def switch_set(self, set_name, current_iter=None):
        self.current_set_name = set_name
        if set_name == "train":
            self.update_seed(dataset_name=set_name, seed=self.init_seed[set_name] + current_iter)

    def update_seed(self, dataset_name, seed=100):
        self.seed[dataset_name] = seed

    def __getitem__(self, idx):

        support_set_images, support_set_augmented_images, target_set_image, target_set_image_augmented, support_set_labels, target_set_label, support_set_original_labels, target_set_original_labels, seed = \
            self.get_set(self.current_set_name, seed=self.seed[self.current_set_name] + idx,
                         augment_images=self.augment_images)

        return support_set_images, support_set_augmented_images, target_set_image, target_set_image_augmented, support_set_labels, target_set_label, support_set_original_labels, target_set_original_labels, seed

    def reset_seed(self):
        self.seed = self.init_seed


class MetaLearningSystemDataLoader(object):
    def __init__(self, args, current_iter=0, model=None):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_iter: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.num_of_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        self.samples_per_iter = args.samples_per_iter
        self.num_workers = args.num_dataprovider_workers
        self.total_train_iters_produced = 0
        self.dataset = FewShotLearningDatasetParallel(args=args, model=model)
        self.batches_per_iter = args.samples_per_iter
        self.full_data_length = self.dataset.data_length
        self.continue_from_iter(current_iter=current_iter)
        self.args = args

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=(self.num_of_gpus * self.batch_size * self.samples_per_iter),
                          shuffle=False, num_workers=self.num_workers, drop_last=True)

    def continue_from_iter(self, current_iter):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_iter:
        """
        self.total_train_iters_produced += (current_iter * (self.num_of_gpus * self.batch_size * self.samples_per_iter))

    def get_train_batches(self, total_batches=-1, augment_images=False):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="train", current_iter=self.total_train_iters_produced)
        self.dataset.set_augmentation(augment_images=augment_images)
        self.total_train_iters_produced += (self.num_of_gpus * self.batch_size * self.samples_per_iter)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

    def get_val_batches(self, total_batches=-1, augment_images=False):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name="val")
        self.dataset.set_augmentation(augment_images=augment_images)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched


    def get_test_batches(self, total_batches=-1, augment_images=False):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches * self.dataset.batch_size
        self.dataset.switch_set(set_name='test')
        self.dataset.set_augmentation(augment_images=augment_images)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

