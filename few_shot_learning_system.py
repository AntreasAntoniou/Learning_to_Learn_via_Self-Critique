import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from meta_neural_network_architectures import VGGActivationNormNetwork, VGGActivationNormNetworkWithAttention
from meta_optimizer import LSLRGradientDescentLearningRule
from standard_neural_network_architectures import CriticNetwork, \
    TaskRelationalEmbedding, \
    SqueezeExciteDenseNetEmbeddingSmallNetwork


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MAMLFewShotClassifier(nn.Module):
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.args.device = device
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = -1
        self.clip_grads = True
        self.rng = set_torch_seed(seed=args.train_seed)
        self.build_module()

    def build_module(self):
        return NotImplementedError

    def setup_optimizer(self, args):

        exclude_param_string = None if "none" in self.args.exclude_param_string else self.args.exclude_param_string
        self.optimizer = optim.Adam(self.trainable_parameters(exclude_params_with_string=exclude_param_string),
                                    lr=args.meta_learning_rate,
                                    weight_decay=args.weight_decay, amsgrad=False)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                              T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)
        self.to(self.device)

        print("Outer Loop parameters")
        for name, param in self.trainable_names_parameters(exclude_params_with_string=exclude_param_string):
            if param.requires_grad:
                print(name, param.shape)

    def get_params_that_include_strings(self, included_strings, include_all=False):
        for name, param in self.named_parameters():
            if any([included_string in name for included_string in included_strings]) and not include_all:
                yield param
            if all([included_string in name for included_string in included_strings]) and include_all:
                yield param

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = self.args.minimum_per_task_contribution / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        self.classifier.zero_grad(params=names_weights_copy)
        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('NOT FOUND', key)

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)

        return names_weights_copy

    def get_inner_loop_parameter_dict(self, params, exclude_strings=None):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()

        if exclude_strings is None:
            exclude_strings = []

        for name, param in params:
            if param.requires_grad:
                if all([item not in name for item in exclude_strings]):
                    if self.args.enable_inner_loop_optimizable_bn_params:
                        param_dict[name] = param.to(device=self.device)
                    else:
                        if "norm_layer" not in name:
                            param_dict[name] = param.to(device=self.device)
        return param_dict

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step,
                    return_features=False):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """

        if return_features:
            preds, features = self.classifier.forward(x=x, params=weights,
                                                      training=training,
                                                      backup_running_statistics=backup_running_statistics,
                                                      num_step=num_step,
                                                      return_features=return_features)

            loss = F.cross_entropy(preds, y)

            return loss, preds, features


        else:
            preds = self.classifier.forward(x=x, params=weights,
                                            training=training,
                                            backup_running_statistics=backup_running_statistics,
                                            num_step=num_step)

            loss = F.cross_entropy(preds, y)

            return loss, preds

    def trainable_parameters(self, exclude_params_with_string=None):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for name, param in self.named_parameters():
            if exclude_params_with_string is not None:
                if param.requires_grad and all(
                        list([exclude_string not in name for exclude_string in exclude_params_with_string])):
                    yield param
            else:
                if param.requires_grad:
                    yield param

    def trainable_names_parameters(self, exclude_params_with_string=None):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for name, param in self.named_parameters():
            if exclude_params_with_string is not None:
                if param.requires_grad and all(
                        list([exclude_string not in name for exclude_string in exclude_params_with_string])):
                    yield (name, param)
            else:
                if param.requires_grad:
                    yield (name, param)

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        if self.args.train_in_stages == True and epoch > self.args.train_learning_rates_as_a_stage_num_epochs:
            self.optimizer = optim.Adam(self.trainable_parameters(), lr=self.args.meta_learning_rate,
                                        weight_decay=self.args.weight_decay, amsgrad=False)

            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                  T_max=self.args.total_epochs,
                                                                  eta_min=self.args.min_learning_rate)

        losses, per_task_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                              use_second_order=self.args.second_order and
                                                               epoch > self.args.first_order_to_second_order_epoch,
                                              use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                              num_steps=self.args.number_of_training_steps_per_iter,
                                              training_phase=True)
        return losses, per_task_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                              use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                              num_steps=self.args.number_of_evaluation_steps_per_iter,
                                              training_phase=False)

        return losses, per_task_preds

    def meta_update(self, loss, exclude_string_list=None, retain_graph=False):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if 'imagenet' in self.args.dataset_name:
            for name, param in self.trainable_names_parameters(exclude_params_with_string=exclude_string_list):
                #

                if self.clip_grads and param.grad is None and param.requires_grad:
                    print(name, 'no grad information computed')
                # else:
                #     print("passed", name)
                else:
                    if param.grad is None:
                        print('no grad information computed', name)
                # print('No Grad', name, param.shape)
                if self.clip_grads and param.grad is not None and param.requires_grad and "softmax":
                    param.grad.data.clamp_(-10, 10)

        self.optimizer.step()


class EmbeddingMAMLFewShotClassifier(MAMLFewShotClassifier):
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(EmbeddingMAMLFewShotClassifier, self).__init__(im_shape=im_shape, device=device, args=args)

        self.args = args
        self.args.device = device
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = -1
        self.rng = set_torch_seed(seed=args.train_seed)

    def build_module(self):
        support_set_shape = (
            self.args.num_classes_per_set * self.args.num_samples_per_class,
            self.args.image_channels,
            self.args.image_height, self.args.image_width)

        target_set_shape = (
            self.args.num_classes_per_set * self.args.num_target_samples,
            self.args.image_channels,
            self.args.image_height, self.args.image_width)

        x_support_set = torch.ones(support_set_shape)
        x_target_set = torch.ones(target_set_shape)

        # task_size = x_target_set.shape[0]
        x_target_set = x_target_set.view(-1, x_target_set.shape[-3], x_target_set.shape[-2], x_target_set.shape[-1])
        x_support_set = x_support_set.view(-1, x_support_set.shape[-3], x_support_set.shape[-2],
                                           x_support_set.shape[-1])

        num_target_samples = x_target_set.shape[0]
        num_support_samples = x_support_set.shape[0]

        self.dense_net_embedding = SqueezeExciteDenseNetEmbeddingSmallNetwork(
            im_shape=torch.cat([x_support_set, x_target_set], dim=0).shape, num_filters=self.args.num_filters,
            num_blocks_per_stage=self.args.num_blocks_per_stage,
            num_stages=self.args.num_stages, average_pool_outputs=False, dropout_rate=self.args.dropout_rate,
            output_spatial_dimensionality=self.args.output_spatial_dimensionality)

        if type(self.device) is list:
            init_device = self.device[0]
        else:
            init_device = self.device

        task_features = self.dense_net_embedding.forward(
            x=torch.cat([x_support_set, x_target_set], dim=0), dropout_training=True)
        task_features = task_features.squeeze()
        encoded_x = task_features
        support_set_features = F.avg_pool2d(encoded_x[:num_support_samples], encoded_x.shape[-1]).squeeze()

        self.current_iter = 0

        self.classifier = VGGActivationNormNetworkWithAttention(args=self.args, im_shape=encoded_x.shape,
                                                                num_output_classes=self.args.num_classes_per_set,
                                                                device=init_device, use_channel_wise_attention=True,
                                                                use_spatial_attention=False, num_stages=1)

        print("init learning rate", self.args.init_learning_rate)
        names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

        task_name_params = self.get_inner_loop_parameter_dict(self.named_parameters())

        if self.args.num_target_set_steps > 0:
            preds, penultimate_features_x = self.classifier.forward(x=encoded_x, num_step=0, return_features=True)
            if 'task_embedding' in self.args.conditional_information:
                self.task_relational_network = TaskRelationalEmbedding(input_shape=support_set_features.shape,
                                                                       num_samples_per_class=self.args.num_samples_per_class,
                                                                       num_classes_per_set=self.args.num_classes_per_set)
                relational_encoding_x = self.task_relational_network.forward(x_img=support_set_features)
                relational_embedding_shape = relational_encoding_x.shape
            else:
                self.task_relational_network = None
                relational_embedding_shape = None

            x_support_set_task = F.avg_pool2d(
                encoded_x[:self.args.num_classes_per_set * (self.args.num_samples_per_class)],
                encoded_x.shape[-1]).squeeze()
            x_target_set_task = F.avg_pool2d(
                encoded_x[self.args.num_classes_per_set * (self.args.num_samples_per_class):],
                encoded_x.shape[-1]).squeeze()
            x_support_set_classifier_features = F.avg_pool2d(penultimate_features_x[
                                                             :self.args.num_classes_per_set * (
                                                                 self.args.num_samples_per_class)],
                                                             penultimate_features_x.shape[-2]).squeeze()
            x_target_set_classifier_features = F.avg_pool2d(
                penultimate_features_x[self.args.num_classes_per_set * (self.args.num_samples_per_class):],
                penultimate_features_x.shape[-2]).squeeze()

            self.critic_network = CriticNetwork(
                task_embedding_shape=relational_embedding_shape,
                num_classes_per_set=self.args.num_classes_per_set,
                support_set_feature_shape=x_support_set_task.shape,
                target_set_feature_shape=x_target_set_task.shape,
                support_set_classifier_pre_last_features=x_support_set_classifier_features.shape,
                target_set_classifier_pre_last_features=x_target_set_classifier_features.shape,

                num_target_samples=self.args.num_target_samples,
                num_samples_per_class=self.args.num_samples_per_class,
                logit_shape=preds[self.args.num_classes_per_set * (self.args.num_samples_per_class):].shape,
                args=self.args, support_set_label_shape=(
                    self.args.num_classes_per_set * (self.args.num_samples_per_class), self.args.num_classes_per_set))

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=init_device,
                                                                    total_num_inner_loop_steps=self.args.num_support_set_steps + self.args.num_target_set_steps,
                                                                    learnable_learning_rates=self.args.learnable_learning_rates,
                                                                    init_learning_rate=self.args.init_learning_rate)

        self.inner_loop_optimizer.initialise(names_weights_dict=names_weights_copy)
        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)
        self.use_cuda = self.args.use_cuda
        self.device = self.device

        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        self.exclude_list = None
        self.switch_opt_params(exclude_list=self.exclude_list)

        if type(self.device) is list:
            self.to(self.device[0])
            self.dense_net_embedding = nn.DataParallel(module=self.dense_net_embedding)
            self.device = self.device[0]
        else:
            self.to(self.device)

    def switch_opt_params(self, exclude_list):
        print("current trainable params")
        for name, param in self.trainable_names_parameters(exclude_params_with_string=exclude_list):
            print(name, param.shape)
        self.optimizer = optim.Adam(self.trainable_parameters(exclude_list), lr=self.args.meta_learning_rate,
                                    weight_decay=self.args.weight_decay, amsgrad=False)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step,
                    return_features=False):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        outputs = {"loss": 0., "preds": 0, "features": 0.}
        if return_features:
            outputs['preds'], outputs['features'] = self.classifier.forward(x=x, params=weights,
                                                                            training=training,
                                                                            backup_running_statistics=backup_running_statistics,
                                                                            num_step=num_step,
                                                                            return_features=return_features)
            if type(outputs['preds']) == tuple:
                if len(outputs['preds']) == 2:
                    outputs['preds'] = outputs['preds'][0]

            outputs['loss'] = F.cross_entropy(outputs['preds'], y)


        else:
            outputs['preds'] = self.classifier.forward(x=x, params=weights,
                                                       training=training,
                                                       backup_running_statistics=backup_running_statistics,
                                                       num_step=num_step)

            if type(outputs['preds']) == tuple:
                if len(outputs['preds']) == 2:
                    outputs['preds'] = outputs['preds'][0]

            outputs['loss'] = F.cross_entropy(outputs['preds'], y)

        return outputs

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """

        x_support_set, x_support_set_augmented, x_target_set, x_target_set_augmented, y_support_set, y_target_set, y_support_set_original, y_target_set_original = data_batch

        self.classifier.zero_grad()

        total_per_step_losses = []

        total_per_step_accuracies = []

        per_task_preds = []
        num_losses = 2
        importance_vector = torch.Tensor([1.0 / num_losses for i in range(num_losses)]).to(self.device)
        step_magnitude = (1.0 / num_losses) / self.args.total_epochs
        current_epoch_step_magnitude = torch.ones(1).to(self.device) * (step_magnitude * (epoch + 1))
        importance_vector[0] = importance_vector[0] - current_epoch_step_magnitude
        importance_vector[1] = importance_vector[1] + current_epoch_step_magnitude
        pre_target_loss_update_loss = []
        pre_target_loss_update_acc = []
        post_target_loss_update_loss = []
        post_target_loss_update_acc = []
        saved_logits_list = None
        num_support_set_steps = self.args.num_support_set_steps
        task_size = x_target_set.shape[0]

        if self.args.use_augmentations == True:
            x_support_set = x_support_set_augmented
            x_target_set = x_target_set_augmented

        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):

            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            c, h, w = x_target_set_task.shape[-3:]
            # in the future try to adapt the features using a relational component
            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            image_embedding = self.dense_net_embedding.forward(
                x=torch.cat([x_support_set_task, x_target_set_task], dim=0), dropout_training=True)

            x_support_set_task = image_embedding[:x_support_set.shape[1] * x_support_set.shape[2]]
            x_target_set_task = image_embedding[x_support_set.shape[1] * x_support_set.shape[2]:]

            if self.args.num_target_set_steps > 0:
                x_support_set_task_features = F.avg_pool2d(x_support_set_task, x_support_set_task.shape[-1]).squeeze()
                x_target_set_task_features = F.avg_pool2d(x_target_set_task,
                                                          x_support_set_task.shape[-1]).squeeze()
                if self.task_relational_network is not None:
                    task_embedding = self.task_relational_network.forward(x_img=x_support_set_task_features)
                else:
                    task_embedding = None
            else:
                task_embedding = None

            for num_step in range(self.args.num_support_set_steps):
                support_outputs = self.net_forward(x=x_support_set_task,
                                                   y=y_support_set_task,
                                                   weights=names_weights_copy,
                                                   backup_running_statistics=
                                                   True if (num_step == 0) else False,
                                                   training=True, num_step=num_step,
                                                   return_features=True)

                names_weights_copy = self.apply_inner_loop_update(loss=support_outputs['loss'],
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

            pre_update_outputs = self.net_forward(x=x_target_set_task,
                                                  y=y_target_set_task, weights=names_weights_copy,
                                                  backup_running_statistics=False, training=True,
                                                  num_step=num_support_set_steps,
                                                  return_features=True)

            if self.args.save_preds:
                if saved_logits_list is None:
                    saved_logits_list = []

                saved_logits_list.extend(pre_update_outputs['preds'])

            y_support_one_hot = torch.zeros(y_support_set_task.shape[0], self.args.num_classes_per_set).to(
                self.device)
            y_support_one_hot.scatter_(1, y_support_set_task.view(-1, 1), 1)

            for num_step in range(self.args.num_target_set_steps):
                if num_step == 0:
                    support_set_features = F.avg_pool2d(support_outputs['features'],
                                                        support_outputs['features'].shape[-1]).squeeze()
                    pre_updated_target_features = F.avg_pool2d(pre_update_outputs['features'],
                                                               pre_update_outputs['features'].shape[-1]).squeeze()

                predicted_loss = self.critic_network.forward(support_set_features=x_support_set_task_features,
                                                             target_set_features=x_target_set_task_features,
                                                             logits=pre_update_outputs['preds'],
                                                             support_set_classifier_pre_last_layer=support_set_features,
                                                             target_set_classifier_pre_last_layer=pre_updated_target_features,
                                                             support_set_labels=y_support_one_hot,
                                                             task_embedding=task_embedding)

                names_weights_copy = self.apply_inner_loop_update(loss=predicted_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)
            if self.args.num_target_set_steps > 0:
                post_update_outputs = self.net_forward(
                    x=x_target_set_task,
                    y=y_target_set_task, weights=names_weights_copy,
                    backup_running_statistics=False, training=True,
                    num_step=num_support_set_steps + 1,
                    return_features=True)
                post_update_loss, post_update_target_preds, post_updated_target_features = post_update_outputs['loss'], \
                                                                                           post_update_outputs['preds'], \
                                                                                           post_update_outputs[
                                                                                               'features']
            else:
                post_update_loss, post_update_target_preds, post_updated_target_features = pre_update_outputs['loss'], \
                                                                                           pre_update_outputs['preds'], \
                                                                                           pre_update_outputs[
                                                                                               'features']

            pre_target_loss_update_loss.append(pre_update_outputs['loss'])
            pre_softmax_target_preds = F.softmax(pre_update_outputs['preds'], dim=1).argmax(dim=1)
            pre_update_accuracy = torch.eq(pre_softmax_target_preds, y_target_set_task).data.cpu().float().mean()
            pre_target_loss_update_acc.append(pre_update_accuracy)

            post_target_loss_update_loss.append(post_update_loss)
            post_softmax_target_preds = F.softmax(post_update_target_preds, dim=1).argmax(dim=1)
            post_update_accuracy = torch.eq(post_softmax_target_preds, y_target_set_task).data.cpu().float().mean()
            post_target_loss_update_acc.append(post_update_accuracy)

            loss = pre_update_outputs['loss'] * importance_vector[0] + post_update_loss * importance_vector[1]

            total_per_step_losses.append(loss)
            total_per_step_accuracies.append(post_update_accuracy)

            per_task_preds.append(post_update_target_preds.detach().cpu().numpy())

            if not training_phase:
                self.classifier.restore_backup_stats()

        loss_metric_dict = dict()
        loss_metric_dict['pre_target_loss_update_loss'] = post_target_loss_update_loss
        loss_metric_dict['pre_target_loss_update_acc'] = pre_target_loss_update_acc
        loss_metric_dict['post_target_loss_update_loss'] = post_target_loss_update_loss
        loss_metric_dict['post_target_loss_update_acc'] = post_target_loss_update_acc

        if self.args.save_preds:
            loss_metric_dict['saved_logits'] = torch.cat(saved_logits_list, dim=0)

        losses = self.get_across_task_loss_metrics(total_losses=total_per_step_losses,
                                                   total_accuracies=total_per_step_accuracies,
                                                   loss_metrics_dict=loss_metric_dict)

        return losses, per_task_preds

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))

        state = torch.load(filepath, map_location='cpu')
        net = dict(state['network'])

        state['network'] = OrderedDict(net)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        self.starting_iter = state['current_iter']

        return state

    def run_train_iter(self, data_batch, epoch, current_iter):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        x_support_set, x_support_set_augmented, x_target_set, x_target_set_augmented, y_support_set, y_target_set, y_support_set_original, y_target_set_original = data_batch

        if type(self.device) is list:
            init_device = self.device[0]
        else:
            init_device = self.device


        x_support_set_augmented = torch.Tensor(x_support_set_augmented).float().to(device=init_device)
        x_support_set = torch.Tensor(x_support_set).float().to(device=init_device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=init_device)
        x_target_set_augmented = torch.Tensor(x_target_set_augmented).float().to(device=init_device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=init_device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=init_device)
        y_support_set_original = torch.Tensor(y_support_set_original).long().to(device=init_device)
        y_target_set_original = torch.Tensor(y_target_set_original).long().to(device=init_device)

        data_batch = (
        x_support_set, x_support_set_augmented, x_target_set, x_target_set_augmented, y_support_set, y_target_set,
        y_support_set_original,
        y_target_set_original)

        losses, per_task_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)
        exclude_string = None

        self.meta_update(loss=losses['loss'], exclude_string_list=exclude_string)
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.zero_grad()

        self.current_iter += 1

        return losses, per_task_preds

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        if type(self.device) is list:
            init_device = self.device[0]
        else:
            init_device = self.device

        x_support_set, x_support_set_augmented, x_target_set, x_target_set_augmented, y_support_set, y_target_set, y_support_set_original, y_target_set_original = data_batch

        x_support_set_augmented = torch.Tensor(x_support_set_augmented).float().to(device=init_device)
        x_support_set = torch.Tensor(x_support_set).float().to(device=init_device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=init_device)
        x_target_set_augmented = torch.Tensor(x_target_set_augmented).float().to(device=init_device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=init_device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=init_device)
        y_support_set_original = torch.Tensor(y_support_set_original).long().to(device=init_device)
        y_target_set_original = torch.Tensor(y_target_set_original).long().to(device=init_device)

        data_batch = (
            x_support_set, x_support_set_augmented, x_target_set, x_target_set_augmented, y_support_set, y_target_set,
            y_support_set_original,
            y_target_set_original)

        losses, per_task_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        return losses, per_task_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)


    def get_across_task_loss_metrics(self, total_losses, total_accuracies, loss_metrics_dict):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses), dim=(0,))

        losses['accuracy'] = torch.mean(torch.stack(total_accuracies), dim=(0,))

        if 'saved_logits' in loss_metrics_dict:
            losses['saved_logits'] = loss_metrics_dict['saved_logits']
            del loss_metrics_dict['saved_logits']

        for name, value in loss_metrics_dict.items():
            losses[name] = torch.stack(value).mean()

        for idx_num_step, (name, learning_rate_num_step) in enumerate(self.inner_loop_optimizer.named_parameters()):
            for idx, learning_rate in enumerate(learning_rate_num_step.mean().view(1)):
                losses['task_learning_rate_num_step_{}_{}'.format(idx_num_step,
                                                                  name)] = learning_rate.detach().cpu().numpy()

        return losses
