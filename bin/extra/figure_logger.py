from copy import deepcopy
import os
import os.path as osp
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


class FigureLogger:
    def __init__(
        self,
        save_dir: str,
        name: str,
        save_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._save_dir = save_dir
        self._name = name
        if save_parameters is None:
            save_parameters = {}
        self.save_parameters = save_parameters

    @property
    def log_dir(self):
        return osp.join(self._save_dir, self._name)

    def initialize_log_dir(self):
        os.makedirs(self.log_dir)

    def log_metrics(
        self,
        figure_name: str,
        metric_values: pd.DataFrame
    ):
        raise NotImplementedError()

    def log_visualisation(
        self,
        figure_name: str,
        figure: plt.Figure,
        save_parameters: Optional[Dict[str, Any]] = None,
    ):
        parameters = deepcopy(self.save_parameters)

        if save_parameters is not None:
            parameters.update(save_parameters)

        figure.savefig(
            osp.join(self.log_dir, figure_name),
            **parameters
        )
