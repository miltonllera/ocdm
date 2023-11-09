import os
import os.path as osp
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union, List

import matplotlib.pyplot as plt
import pandas as pd


class FolderLogger:
    def __init__(
        self,
        save_dir: str,
        name: str,
        save_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if save_parameters is None:
            save_parameters = {}

        self.save_dir = save_dir
        self.name = name
        self.save_parameters = save_parameters

        self.initialize_log_dir()

    @property
    def log_dir(self):
        return osp.join(self.save_dir, self.name)

    def initialize_log_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def log_metrics(
        self,
        figure_name: str,
        metric_values: Union[pd.DataFrame, pd.Series],
    ):
        metric_values.to_csv(osp.join(self.log_dir, f"{figure_name}.csv"))

    def log_visualisation(
        self,
        figure: Union[Tuple[str, plt.Figure], List[Tuple[str, plt.Figure]]],
        save_parameters: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(figure, tuple):
            figure = [figure]

        parameters = deepcopy(self.save_parameters)
        if save_parameters is not None:
            parameters.update(save_parameters)

        for name, fig in figure:
            fig.savefig(
                osp.join(self.log_dir, name),
                **parameters
            )
