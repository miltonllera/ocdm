import re
import numpy as np


def build_filter(dataset, expr: str):
    """
    Build a lambda that evaluates a boolean expression over feature vectors.

    Example:
        build_lambda("shape == 1 & color != 2", ["shape", "color"])
        → lambda x: ( x[:, 0] == 1 ) & ( x[:, 1] != 2 )
    """
    assert hasattr(dataset, 'factors')
    properties = dataset.factors
    prop_to_idx = {p: i for i, p in enumerate(properties)}

    # Pattern to find property names as whole words
    pattern = r"\b(" + "|".join(map(re.escape, properties)) + r")\b"

    # Replace each property with x[index]
    def replace_prop(match):
        prop = match.group(1)
        return f"x[:, {prop_to_idx[prop]}]"

    transformed_expr = re.sub(pattern, replace_prop, expr)

    safe_env = {"np": np}
    lambda_str = f"lambda x: {transformed_expr}"
    # print(lambda_str)
    # exit()
    func = eval(lambda_str, safe_env)

    return func


# if __name__ == "__main__":
#     from itertools import product

#     class dummy_dataset:
#         factors = ('shape', 'scale')

#         def __init__(self) -> None:
#             shape = np.array([1., 2., 3.])
#             scale = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.])
#             self.factor_values = np.asarray(list(product(shape, scale)))


#     data = dummy_dataset()
#     print(data.factor_values)
#     filter_fn = build_filter(data, "( shape ==  1 ) & ( scale > 0.6 )")
#     print(filter_fn(data.factor_values))

