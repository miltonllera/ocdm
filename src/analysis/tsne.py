import numpy as np
from fitsne import FItSNE as run_fitsne


class FItSNE:
    """
    FFT-accelerated Interpolation-based t-SNE as defined in:

        https://www.nature.com/articles/s41592-018-0308-4

    The underlying implementation's repository can be found here:

        https://github.com/KlugerLab/FIt-SNE

    Additionally, they suggest several other articles which propose different
    parameter combinations that help the algorithm converge faster while also
    discovering better substructures in the dataset. These include:

        1. Early exageration: https://epubs.siam.org/doi/abs/10.1137/18M1216134
        2. Late exageration: https://www.nature.com/articles/s41467-019-13056-x
        3. Initialization: https://www.nature.com/articles/s41467-019-13056-x
        4. Variable degrees of freedom: https://ecmlpkdd2019.org/downloads/paper/327.pdf
        5. Perplexity combination: https://www.nature.com/articles/s41467-019-13056-x
        6. Learning rate size: https://www.nature.com/articles/s41467-019-13055-y

    """
    def __init__(
        self,
        no_dims: int=2,
        perplexity: float=30.0,
        sigma: float=-30.0,
        K: int=-1,
        initialization = 'pca',
        load_affinities = None,
        perplexity_list=None,
        theta: float=0.5,
        rand_seed: int=-1,
        max_iter: int=750,
        stop_early_exag_iter: int=250,
        fft_not_bh: bool=True,
        ann_not_vptree: bool=True,
        early_exag_coeff: float=12.0,
        no_momentum_during_exag: bool=False,
        start_late_exag_iter: int=-1,
        late_exag_coeff: float=-1,
        mom_switch_iter: int=250,
        momentum: float=0.5,
        final_momentum: float=0.8,
        learning_rate='auto',
        max_step_norm: float=5,
        n_trees: int=50,
        search_k: int=None,
        nterms: int=3,
        intervals_per_integer: float=1,
        min_num_intervals: int=50,
        nthreads:  int=0,
        df: float=1.0,
    ) -> None:
        self.no_dims = no_dims
        self.perplexity = perplexity
        self.sigma = sigma
        self.K = K
        self.initialization = initialization
        self.load_affinities = load_affinities
        self.perplexity_list = perplexity_list
        self.theta = theta
        self.rand_seed = rand_seed
        self.max_iter = max_iter
        self.stop_early_exag_iter = stop_early_exag_iter
        self.fft_not_bh = fft_not_bh
        self.ann_not_vptree = ann_not_vptree
        self.early_exag_coeff = early_exag_coeff
        self.no_momentum_during_exag = no_momentum_during_exag
        self.start_late_exag_iter = start_late_exag_iter
        self.late_exag_coeff = late_exag_coeff
        self.mom_switch_iter = mom_switch_iter
        self.momentum = momentum
        self.final_momentum = final_momentum
        self.learning_rate = learning_rate
        self.max_step_norm = max_step_norm
        self.n_trees = n_trees
        self.search_k = search_k
        self.nterms = nterms
        self.intervals_per_integer = intervals_per_integer
        self.min_num_intervals = min_num_intervals
        self.nthreads = nthreads
        self.df = df

    def fit_transform(self, X: np.ndarray):
        return run_fitsne(
            X.astype(np.double),
            self.no_dims,
            self.perplexity,
            self.sigma,
            self.K,
            self.initialization,
            self.load_affinities,
            self.perplexity_list,
            self.theta,
            self.rand_seed,
            self.max_iter,
            self.stop_early_exag_iter,
            self.fft_not_bh,
            self.ann_not_vptree,
            self.early_exag_coeff,
            self.no_momentum_during_exag,
            self.start_late_exag_iter,
            self.late_exag_coeff,
            self.mom_switch_iter,
            self.momentum,
            self.final_momentum,
            self.learning_rate,
            self.max_step_norm,
            self.n_trees,
            self.search_k,
            self.nterms,
            self.intervals_per_integer,
            self.min_num_intervals,
            self.nthreads,
            self.df,
        )
