"""
CLI alias for scripts.analyze_latents.
Provides command-line interface for latent representation analysis using PCA and UMAP.
"""
from scripts.analyze_latents import main, parser

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
