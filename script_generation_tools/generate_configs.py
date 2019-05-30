import os
from collections import namedtuple

seed_list = [0, 1, 2]# 3, 4]

config = namedtuple('config', 'dataset_name num_classes '
                              'samples_per_class '
                              'target_samples_per_class exclude_param_string weight_decay num_target_set_steps '
                              'batch_size inner_loop_optimizer_type conditional_information '
                              'train_update_steps '
                              'val_update_steps total_epochs output_spatial_dimensionality '
                              'inner_loop_learning_rate experiment_name num_filters conv_padding load_into_memory learnable_bn_gamma learnable_bn_beta num_stages num_blocks_per_stage learnable_learning_rates learnable_betas ')


def string_generator(string_list):
    output_string = "["
    i = 0
    for string_entry in string_list:
        if i < len(string_list) - 1:
            output_string += "\"{}\",".format(string_entry)
        else:
            output_string += "\"{}\"".format(string_entry)

        i += 1

    output_string += "]"
    return output_string

experiment_conditional_information_config = [["preds"],
                                             ["task_embedding", "preds"]]
# "task_embedding", "penultimate_layer_features"]
configs_list = []


for (k_shot, batch_size) in [(1, 2), (5, 1)]:
    for output_dim in [20]:
            configs_list.append(
                config(dataset_name="mini_imagenet", num_classes=5, samples_per_class=k_shot,
                       target_samples_per_class=15,
                       batch_size=batch_size, train_update_steps=5, val_update_steps=5, inner_loop_learning_rate=0.01,
                       conv_padding=1, num_filters=48, load_into_memory=True,
                       conditional_information=[],
                       num_target_set_steps=0, weight_decay=0.0,
                       total_epochs=150,
                       exclude_param_string=string_generator(
                           ["None"]),
                       experiment_name='standard_5_way_{}_shot_48_{}_{}_LSLR_conditioned'.format(
                           k_shot, "_".join([],), output_dim), learnable_bn_beta=True,
                       learnable_bn_gamma=True,
                       num_stages=4, num_blocks_per_stage=0,
                       inner_loop_optimizer_type='LSLR', learnable_betas=False, learnable_learning_rates=True,
                       output_spatial_dimensionality=output_dim))
            for conditional_information in experiment_conditional_information_config:
                configs_list.append(
                    config(dataset_name="mini_imagenet", num_classes=5, samples_per_class=k_shot,
                           target_samples_per_class=15,
                           batch_size=batch_size, train_update_steps=5, val_update_steps=5, inner_loop_learning_rate=0.01,
                           conv_padding=1, num_filters=48, load_into_memory=True,
                           conditional_information=string_generator(conditional_information),
                           num_target_set_steps=1, weight_decay=0.0,
                           total_epochs=150,
                           exclude_param_string=string_generator(
                               ["None"]),
                           experiment_name='intrinsic_5_way_{}_shot_48_{}_{}_LSLR_conditioned'.format(
                               k_shot, "_".join(conditional_information), output_dim), learnable_bn_beta=True,
                           learnable_bn_gamma=True,
                           num_stages=4, num_blocks_per_stage=0,
                           inner_loop_optimizer_type='LSLR', learnable_betas=False, learnable_learning_rates=True,
                           output_spatial_dimensionality=output_dim))


experiment_templates_json_dir = '../experiment_template_config/'
experiment_config_target_json_dir = '../experiment_config/'

if not os.path.exists(experiment_config_target_json_dir):
    os.makedirs(experiment_config_target_json_dir)


def fill_template(script_text, config):
    for key, value in config.items():
        script_text = script_text.replace('${}$'.format(key), str(value).lower())

    return script_text


def load_template(filepath):
    with open(filepath, mode='r') as filereader:
        template = filereader.read()

    return template


def write_text_to_file(text, filepath):
    with open(filepath, mode='w') as filewrite:
        filewrite.write(text)


for subdir, dir, files in os.walk(experiment_templates_json_dir):
    for template_file in files:
        for seed_idx in seed_list:
            filepath = os.path.join(subdir, template_file)

            for config in configs_list:
                loaded_template_file = load_template(filepath=filepath)
                config_dict = config._asdict()
                config_dict['train_seed'] = seed_idx
                config_dict['val_seed'] = seed_idx
                config_dict['experiment_name'] = "{}_{}_{}".format(template_file.replace(".json", ''),
                                                                   config.experiment_name, seed_idx)
                cluster_script_text = fill_template(script_text=loaded_template_file,
                                                    config=config_dict)

                cluster_script_name = '{}/{}_{}_{}.json'.format(experiment_config_target_json_dir,
                                                                template_file.replace(".json", ''),
                                                                config.experiment_name, seed_idx)
                print(cluster_script_name, seed_idx)
                cluster_script_name = os.path.abspath(cluster_script_name)
                write_text_to_file(cluster_script_text, filepath=cluster_script_name)
