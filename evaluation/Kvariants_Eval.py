# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import json
import torch
import random
import numpy as np
from loader.LoadData import load_data
from utils import Logger


class KVariantEval:

    def __init__(self, data, exp_path, model_config):
        self.num_cls = data.num_cls
        self.data = data.data
        self.model_config = model_config
        self._NESTED_FOLDER = exp_path
        self._FOLD_BASE = '_CLS'
        self._RESULTS_FILENAME = 'results.json'
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def risk_assessment(self, experiment_class):

        if not os.path.exists(self._NESTED_FOLDER):
            os.makedirs(self._NESTED_FOLDER)

        for cls in range(self.num_cls):

            folder = os.path.join(self._NESTED_FOLDER, str(cls)+self._FOLD_BASE)
            if not os.path.exists(folder):
                os.makedirs(folder)

            json_results = os.path.join(folder, self._RESULTS_FILENAME)
            if not os.path.exists(json_results):

                self._risk_assessment_helper(cls, experiment_class, folder)
            else:
                print(
                    f"File {json_results} already present! Shutting down to prevent loss of previous experiments")
                continue

    def _risk_assessment_helper(self, cls, experiment_class, exp_path):

        config = self.model_config
        experiment = experiment_class(config, exp_path)

        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')
        # logger = None

        num_repeat = config['num_repeat']
        saved_results = {}
        # Mitigate bad random initializations
        for i in range(num_repeat):
            torch.cuda.empty_cache()
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(i + 42)
            random.seed(i + 42)
            torch.manual_seed(i + 42)
            torch.cuda.manual_seed(i + 42)
            torch.cuda.manual_seed_all(i + 42)
            dataset = load_data(self.data, cls)
            scores = experiment.run_test(dataset,logger)
            saved_results['scores_'+str(i)] = scores.tolist()
        if config['save_scores']:
            save_path = os.path.join(self._NESTED_FOLDER, str(cls)+self._FOLD_BASE,'scores_labels.json')
            json.dump(saved_results, open(save_path, 'w'))
            # saved_results = json.load(open(save_path))


