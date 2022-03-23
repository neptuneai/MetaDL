# coding=utf-8
import random
import numpy as np
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, num_classes, in_features, mid_features):
        super(MLPClassifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=mid_features, bias=True),
            nn.GELU(),
            nn.Linear(in_features=mid_features, out_features=mid_features, bias=True),
            nn.GELU(),
            nn.Linear(in_features=mid_features, out_features=num_classes, bias=True),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


class MLPClassifier2(nn.Module):
    def __init__(self, num_classes, in_features, mid_features):
        super(MLPClassifier2, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=mid_features, bias=True),
            nn.GELU(),
            nn.Linear(in_features=mid_features, out_features=mid_features, bias=True),
            nn.GELU(),
            nn.Linear(in_features=mid_features, out_features=num_classes, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


class MLPTrainer(object):
    def __init__(self, criterion=None):
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train_once(self, net, x_data, y_data, lr, epochs):
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.000001)

        net.train()

        min_loss = np.Infinity
        min_epoch = 0
        batch_size = 128
        for epoch in range(0, epochs):

            num_batch = (y_data.shape[0] + batch_size - 1) // batch_size
            for step in range(num_batch):
                index1 = step * batch_size
                index2 = step * batch_size + batch_size

                x_data_b = torch.tensor(x_data[index1:index2, :]).to(self.device).float()
                if y_data.ndim == 2:
                    y_data_b = torch.tensor(y_data[index1:index2, :]).to(self.device).float()
                else:
                    y_data_b = torch.tensor(y_data[index1:index2]).to(self.device)

                outs = net(x_data_b)

                optimizer.zero_grad()

                loss = self.criterion(outs, y_data_b)

                # print('==>train_net, Train epoch: {}, loss: {}'.format(epoch, loss.item()))

                if np.isnan(loss.item()):
                    break

                if loss.item() < min_loss:
                    min_loss = loss.item()
                    min_epoch = epoch
                # else:
                #     break

                loss.backward()
                optimizer.step()
            scheduler.step()

        print('====> train end, lr: {}, loss: {}, epoch: {}'.format(lr, min_loss, min_epoch))
        return min_loss

    def eval_net(self, net, x_data):
        net.eval()

        with torch.no_grad():
            x_data = torch.tensor(x_data).to(self.device).float().unsqueeze(0)
            outs = net(x_data)
        return outs.detach().cpu().numpy()[0]


def key_value_mapping(key, value):
    key_value_dict = {
        'target_type': {'Categorical': 0.0, 'Binary': 1.0, 'Numerical': 2.0},
        'task': {'regression': 0.0, 'multilabel.classification': 1.0, 'multiclass.classification': 2.0,
                 'binary.classification': 3.0},
        'feat_type': {'Mixed': 0.0, 'Binary': 1.0, 'Numerical': 2.0},
        'metric': {'auc_metric': 0.0, 'f1_metric': 1.0, 'bac_metric': 2.0, 'r2_metric': 3.0, 'a_metric': 4.0,
                   'pac_metric': 5.0}
    }

    if key in key_value_dict:
        value = key_value_dict[key][value]
    return float(value)


def draw_curves(x_values_list, y_values_list, name_list, output_file):
    import matplotlib.pyplot as plt

    plt.figure()

    for x_values, y_values, name in zip(x_values_list, y_values_list, name_list):
        plt.plot(x_values, y_values, label=name)

    plt.legend()
    plt.savefig(output_file)


def timestamp_mapping(value):
    value = float(value)
    if value < 10:
        return 0
    elif value < 100:
        return 1
    else:
        return 2


class Agent(object):
    def __init__(self, number_of_algorithms):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        self.algo_name_list = [str(x) for x in range(number_of_algorithms)]
        self.validation_last_scores = [0.0 for _ in range(number_of_algorithms)]

        self.explore_rate = 0.3
        self.total_budget = 500
        self.time_used = 0.0
        self.last_delta_t = 0.5
        self.ds_feat_keys = [
            'target_type',
            'task',
            'feat_type',
            'metric',
            'feat_num',
            'target_num', 'label_num',
            'train_num',
            'valid_num', 'test_num',
            'has_categorical', 'has_missing',
            'is_sparse',
            'time_budget'
        ]
        self.ds_feat_max_values = {key: 1.0 for key in self.ds_feat_keys}
        self.best_algo_list = []
        self.current_dataset_meta_features = None
        self.last_index = 0
        self.trainer = MLPTrainer()
        self.trainer2 = MLPTrainer(criterion=nn.CrossEntropyLoss())
        self.net1 = None
        self.net2 = None
        self.new_timestamps = [
            0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0
        ]

        self.new_timestamps2 = [
            0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0
        ]
        self.ts_labels = [0 for _ in range(number_of_algorithms)]
        self.hp_keys1 = ['meta_feature_0']
        self.hp_keys2 = ['meta_feature_0']

    def _process_learning_curve(self, scores, times, budget):
        # new_times = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
        new_times = self.new_timestamps

        scores = [max(0.0, float(x)) for x in scores]
        times = [float(x) / budget for x in times]

        scores = scores + [scores[-1]]
        times = times + [1.1]
        new_scores = []
        for t in new_times:
            if t < times[0]:
                new_s = 0.0
            else:
                j = 0
                while j < len(times) - 1:
                    if times[j] <= t < times[j + 1]:
                        break
                    j += 1
                delta = (t - times[j]) / (times[j + 1] - times[j])
                new_s = scores[j] + (scores[j + 1] - scores[j]) * delta
                new_s = round(new_s, 4)
            new_scores.append(new_s)
        return new_scores, new_times

    def _get_dataset_vector(self, dataset_meta_features):
        values = []
        for k in self.ds_feat_keys:
            v = key_value_mapping(k, dataset_meta_features[k]) / self.ds_feat_max_values[k]
            values.append(round(v, 6))
        return values

    def reset(self, dataset_meta_features, algorithms_meta_features):
        """ dataset_meta_features
                {'usage': 'Meta-learningchallenge2022', 'name': 'Erik', 'task': 'regression',
                'target_type': 'Binary', 'feat_type': 'Mixed', 'metric': 'f1_metric',
                'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
                'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
                'has_missing': '0', 'is_sparse': '1'}
            algorithms_meta_features
                {'0': {'meta_feature_0': '0', 'meta_feature_1': '0.1'},
                 '1': {'meta_feature_0': '1', 'meta_feature_1': '0.2'},
                 '2': {'meta_feature_0': '0', 'meta_feature_1': '0.3'},
                 '3': {'meta_feature_0': '1', 'meta_feature_1': '0.4'},
                 ...
                 '18': {'meta_feature_0': '1', 'meta_feature_1': '0.9'},
                 '19': {'meta_feature_0': '0', 'meta_feature_1': '1.0'}}
        """

        self.validation_last_scores = [0.0 for _ in self.algo_name_list]
        self.total_budget = float(dataset_meta_features['time_budget'])
        self.time_used = 0.0

        self.current_dataset_meta_features = dataset_meta_features

        self.last_delta_t = 0.1
        self.last_index = 0

        ds_vector = self._get_dataset_vector(dataset_meta_features)

        scores_list = []
        self.ts_labels = [0 for _ in self.algo_name_list]
        for algo_name in self.algo_name_list:
            algo_vector1 = [float(algorithms_meta_features[algo_name][k]) for k in self.hp_keys1]
            algo_vector2 = [float(algorithms_meta_features[algo_name][k]) for k in self.hp_keys2]
            scores = self.trainer.eval_net(self.net1, ds_vector + algo_vector1)
            ts_label = self.trainer.eval_net(self.net2, ds_vector + algo_vector2)

            scores_list.append(scores.tolist())
            self.ts_labels[int(algo_name)] = np.argmax(ts_label)

        self.best_algo_list = []
        for i in range(len(self.new_timestamps)):
            scores = [x[i] for x in scores_list]
            best_idx = np.argmax(scores)
            self.best_algo_list.append(best_idx)

        self.best_algo_list += [0, 10]
        self.best_algo_list = list(set(self.best_algo_list))
        print('===> reset, best algo', dataset_meta_features['name'], self.best_algo_list)
        print('===> reset, best algo', dataset_meta_features['name'], self.ts_labels)

        self.delta_index_list = [0 for _ in self.algo_name_list]
        self.delta_index_list2 = [0 for _ in self.algo_name_list]
        self.algo_index = 0

    def meta_train(self, datasets_meta_features, algorithms_meta_features,
                   validation_learning_curves, test_learning_curves):
        """
        validation_learning_curves['Erik']['0'] <learning_curve.Learning_Curve object at 0x9kwq10eb49a0>
        validation_learning_curves['Erik']['0'].timestamps [196, 319, 334, 374, 409]
        validation_learning_curves['Erik']['0'].scores [0.6465, 0.6465, 0.6465, 0.64652, 0.64652]
        """
        in_features1 = len(self.ds_feat_keys) + len(self.hp_keys1)
        in_features2 = len(self.ds_feat_keys) + len(self.hp_keys2)
        self.net1 = MLPClassifier(num_classes=len(self.new_timestamps), in_features=in_features1, mid_features=128)
        self.net1 = self.net1.to(self.trainer.device)

        self.net2 = MLPClassifier2(num_classes=3, in_features=in_features2, mid_features=32)
        self.net2 = self.net2.to(self.trainer.device)

        algo_hp_value_dict1 = {}
        algo_hp_value_dict2 = {}
        for algo_name in self.algo_name_list:
            algo_hp_value_dict1[algo_name] = [float(algorithms_meta_features[algo_name][k]) for k in self.hp_keys1]
            algo_hp_value_dict2[algo_name] = [float(algorithms_meta_features[algo_name][k]) for k in self.hp_keys2]

        self.ds_feat_max_values = {key: 1.0 for key in self.ds_feat_keys}
        for _, ds in datasets_meta_features.items():
            for key in self.ds_feat_keys:
                v = key_value_mapping(key, ds[key])
                self.ds_feat_max_values[key] = max(self.ds_feat_max_values[key], v)

        ds_vec_dict = {}
        for key, ds in datasets_meta_features.items():
            ds_vec_dict[key] = self._get_dataset_vector(ds)

        inputs_list1 = []
        inputs_list2 = []
        score_list = []
        ts_label_list = []
        ds_name_list = []
        for key, ds in validation_learning_curves.items():
            ds_vector = ds_vec_dict[key]
            time_budget = float(datasets_meta_features[key]['time_budget'])
            ds_name_list.append(datasets_meta_features[key]['name'])
            for algo_name in self.algo_name_list:
                inputs_list1.append(ds_vector + algo_hp_value_dict1[algo_name])
                inputs_list2.append(ds_vector + algo_hp_value_dict2[algo_name])
                new_scores, _ = self._process_learning_curve(ds[algo_name].scores, ds[algo_name].timestamps, time_budget)
                score_list.append(new_scores)

                ts_label_list.append(timestamp_mapping(ds[algo_name].timestamps[0]))

        x1_data = np.array(inputs_list1)
        x2_data = np.array(inputs_list2)
        y1_data = np.array(score_list)
        y2_data = np.array(ts_label_list)

        self.trainer.train_once(self.net1, x1_data, y1_data, lr=0.02, epochs=1000)
        self.trainer2.train_once(self.net2, x2_data, y2_data, lr=0.02, epochs=1000)

    def suggest(self, observation):
        if observation is not None:
            A, C_A, R_validation_C_A = observation
            self.validation_last_scores[A] = max(self.validation_last_scores[A], R_validation_C_A)
        best_algo_for_test = np.argmax(self.validation_last_scores)

        new_times1 = [0, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10, 20, 30, 50, 80, 120, 180, 400, 1000, self.total_budget+10]
        if self.time_used < 250 or max(self.validation_last_scores) <= 0.1:
            if self.algo_index >= len(self.best_algo_list):
                self.algo_index = 0

            next_algo_to_reveal = self.best_algo_list[self.algo_index]
            delta_index = self.delta_index_list[next_algo_to_reveal]
            if delta_index + 2 < len(new_times1):
                if self.ts_labels[next_algo_to_reveal] == 2:
                    delta_t = new_times1[delta_index + 2] - new_times1[delta_index]
                    self.delta_index_list[next_algo_to_reveal] += 2
                else:
                    delta_t = new_times1[delta_index + 1] - new_times1[delta_index]
                    self.delta_index_list[next_algo_to_reveal] += 1
            else:
                delta_t = 50

            self.algo_index += 1
        else:
            next_algo_to_reveal = best_algo_for_test
            delta_t = 50 * (self.ts_labels[next_algo_to_reveal] * 0.5 + 1)

        self.time_used += delta_t
        action = (best_algo_for_test, next_algo_to_reveal, delta_t)
        return action

