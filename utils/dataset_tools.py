import os
import shutil


def maybe_unzip_dataset(args):
    datasets = [args.dataset_name]
    dataset_paths = [args.dataset_path]

    for dataset_idx, dataset_path in enumerate(dataset_paths):
        if dataset_path.endswith('/'):
            dataset_path = dataset_path[:-1]

        shutil.rmtree(dataset_path)

        if not os.path.exists(dataset_path):
            print("Not found dataset folder structure.. searching for .tar.bz2 file")
            zip_directory = "{}.tar.bz2".format(os.path.join(os.environ['DATASET_DIR'], datasets[dataset_idx]))

            assert os.path.exists(
                zip_directory), "dataset zip file not found, please download from gdrive https://drive.google.com/open?id=1ljP5AaiwZoS6LmEx6UquG_UScUaUd4-m and " \
                                "place in datasets folder as explained in README"
            print("Found zip file, unpacking")

            unzip_file(
                filepath_pack=os.path.join(os.environ['DATASET_DIR'], "{}.tar.bz2".format(datasets[dataset_idx])),
                filepath_to_store=os.environ['DATASET_DIR'])
            args.reset_stored_filepaths = True


def unzip_file(filepath_pack, filepath_to_store):
    print("unzipping", filepath_pack, "to", filepath_to_store)
    command_to_run = "tar -I pbzip2 -xf {} -C {}".format(filepath_pack, filepath_to_store)
    os.system(command_to_run)


def check_download_dataset(args):
    datasets = [args.dataset_name]
    dataset_paths = [args.dataset_path]
    print(os.path.abspath(os.environ['DATASET_DIR']))
    done = False
    for dataset_idx, dataset_path in enumerate(dataset_paths):
        if dataset_path.endswith('/'):
            dataset_path = dataset_path[:-1]

        zip_directory = "{}.tar.bz2".format(os.path.join(os.environ['DATASET_DIR'], datasets[dataset_idx]))
        if not os.path.exists(zip_directory):
            print("New dataset not found, resetting")
            shutil.rmtree(dataset_path, ignore_errors=True)

        if not os.path.exists(os.environ['DATASET_DIR']):
            os.mkdir(os.environ['DATASET_DIR'])

        if not os.path.exists(dataset_path):
            print("Not found dataset folder structure.. searching for .tar.bz2 file")
            zip_directory = "{}.tar.bz2".format(os.path.join(os.environ['DATASET_DIR'], datasets[dataset_idx]))
            assert os.path.exists(
                zip_directory), "Dataset zip file can't be found either, please place the dataset zip file or folder in " \
                                "the datasets folder. For more info on how to download Mini-ImageNet and CUB please " \
                                "read the README.md file"

            unzip_file(
                filepath_pack=os.path.join(os.environ['DATASET_DIR'], "{}.tar.bz2".format(datasets[dataset_idx])),
                filepath_to_store=os.environ['DATASET_DIR'])
            args.reset_stored_filepaths = True

        total_files = 0
        for subdir, dir, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(".jpeg") or file.lower().endswith(".jpg") or file.lower().endswith(
                        ".png") or file.lower().endswith(".pkl"):
                    total_files += 1
        print("count stuff________________________________________", total_files)
        if (total_files == 1623 * 20 and datasets[dataset_idx] == 'omniglot_dataset') or (
                total_files == 100 * 600 and 'mini_imagenet' in datasets[dataset_idx]) or (
                total_files == 3 and 'mini_imagenet_pkl' in datasets[dataset_idx]) or (
                total_files == 11788 and "cub" in datasets[dataset_idx]):
            print("file count is correct")
            done = True
        else:
            print("file count is wrong, please delete folder structure and place original .tar.bz2 file in datasets")
            assert os.path.exists(
                zip_directory), "Dataset zip file can't be found either, please place the dataset zip file or folder in " \
                                "the datasets folder. For more info on how to download Mini-ImageNet and CUB please " \
                                "read the README.md file"

            print("Pack is downloaded, unpacking")
            unzip_file(
                filepath_pack=os.path.join(os.environ['DATASET_DIR'], "{}.tar.bz2".format(datasets[dataset_idx])),
                filepath_to_store=os.environ['DATASET_DIR'])
            args.reset_stored_filepaths = True

        if not done:
            check_download_dataset(args=args)
