from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    MaskedAffineAutoregressiveTransform,
    ReversePermutation
)

def build_conditional_maf(context_dim, n_blocks=4, hidden_features=64):
    transforms = []
    for _ in range(n_blocks):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=1,
                hidden_features=hidden_features,
                context_features=context_dim,
                num_blocks=2,
                use_residual_blocks=False
            )
        )
        transforms.append(ReversePermutation(features=1))

    return Flow(
        CompositeTransform(transforms),
        StandardNormal([1])
    )
