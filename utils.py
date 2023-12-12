from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val

    return d() if isfunction(d) else d


def identity(t):
    return t


def extract(tensor, t, x_shape):
    """
    Extract value from tensor by giving t
    :param tensor: e.g: sqrt_alphas_cumprod = tensor([0.9999, 0.9966, 0.9899, 0.9799])
    :param t: e.g, t = tensor([1, 3])
    :param x_shape: e.g, x_start.shape = (b, c, h, w)
    :return:
    """
    # comments below are according to the given example
    batch_size = t.shape[0]             # 2
    out = tensor.gather(-1, t.cpu())    # gather(dim, index), dim=-1, index=(1, 3) => out = tensor([0.9966, 0.9799])
    return out.reshape(
        batch_size,
        *((1,) * (len(x_shape) - 1))    # (1,) * (4-1) = (1, 1, 1) -> *(1, 1, 1)=1, 1, 1
    ).to(t.device)                      # so, out.shape = (2, 1, 1, 1)
    # out:
    # tensor(
    #   [
    #       [[[0.9966]]],
    #       [[[0.9799]]]
    #   ]
    # )




