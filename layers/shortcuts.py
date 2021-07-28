import layers


def c7s1k(k: int, activation: str = "relu") -> layers.ConvolutionBlock:
    """Shortcut function to c7s1-k layer."""
    return layers.ConvolutionBlock(7, 1, k, activation=activation)


def dk(k: int) -> layers.ConvolutionBlock:
    """Shortcut to dk layer.  Reflection padding seems to have only been done on this layer."""
    return layers.ConvolutionBlock(3, 2, k, reflect_padding=True)


def uk(k: int) -> layers.ConvolutionBlock:
    """Shortcut to uk layer."""
    return layers.ConvolutionBlock(3, 2, k, transpose=True)


def ck(k: int, normalize: bool = True) -> layers.ConvolutionBlock:
    """Shortcut to ck layer."""
    return layers.ConvolutionBlock(4, 2, k, normalize=normalize, leaky_slope=0.2)


def rk(k: int, filters_changed: bool = False) -> layers.ResidualBlock:
    """Shortcut to residual blocks.  It's undefined in the paper what their stride is so we'll assume 1."""
    return layers.ResidualBlock(n_filters=k, stride=1, change_n_channels=filters_changed)
