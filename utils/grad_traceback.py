import copy
import math
import torch
import torch.nn as nn


def filter(x, dim=1, scale=1.0, target_shape=None, model_name=None, split=0):
    """
    Filters out tensor elements based on a scaling factor or target shape.

    Args:
        x: Input tensor.
        dim: Dimension to filter along.
        scale: Fraction of elements to retain.
        target_shape: Desired shape to adjust to.
        model_name: Name of the model to handle special cases.
        split: Number of splits for selective filtering.

    Returns:
        Tensor with filtered elements set to 0.(True means filter set to 0)
    """
    if target_shape is None:
        if scale == 1:
            return x
        out_dim = int(scale * x.shape[dim])
    else:
        if dim == 1:
            out_dim = target_shape[0]
        else:
            out_dim = target_shape[0] if model_name == 'Linear' else target_shape[-1]

    # Create a mask to filter out unwanted elements
    ind_list = [slice(None)] * len(x.shape)  # Initialize index list
    if split:
        total_dim = x.shape[min(dim, x.ndim - 1)] # out_channel
        keep = [False] * (out_dim // split) + [True] * (total_dim // split - out_dim // split)
        ind_list[min(dim, x.ndim - 1)] = keep * split
    else:
        ind_list[min(dim, x.ndim - 1)] = slice(out_dim, x.shape[min(dim, x.ndim - 1)])
    x[tuple(ind_list)] = 0  # Mask the filtered elements

    return x


def get_num_gen(gen):
    """Counts the number of items in a generator."""
    return sum(1 for _ in gen)


def is_leaf(model):
    """Checks if a model is a leaf node (no children)."""
    return get_num_gen(model.children()) == 0


def get_downscale_index(model, args, scale=1.0):
    """
    Computes the index of parameters to retain during downscaling.

    Args:
        model: PyTorch model.
        args: Arguments containing configuration.
        scale: Scaling factor for downscaling.

    Returns:
        Dictionary mapping parameter names to retention masks. (True means have gradient)
    """
    dim = 1  # Default dimension for filtering

    def should_filter(layer):
        """Checks if a layer should be filtered."""
        return is_leaf(layer) and not hasattr(layer, 'modify_forward')

    def restore_forward(model):
        """Restores the original forward functions of all layers."""
        for child in model.children():
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
            else:
                restore_forward(child)

    def modify_forward(model, local_model, split=1):
        """
        Modifies the forward pass of the model to apply filtering.

        Args:
            model: Original model.
            local_model: Local copy of the model.
            split: Number of splits for filtering.
        """
        for i, child in enumerate(model.children()):
            local_child = list(local_model.children())[i]

            if should_filter(child):
                def new_forward(layer, target_shape):
                    def lambda_forward(x):
                        # Handle normalization and activation layers without affecting them
                        if 'BatchNorm' in layer._get_name():
                            x_ = x + layer.weight[None, :, None, None] + layer.bias[None, :, None, None]
                        elif 'LayerNorm' in layer._get_name():
                            x_ = x + layer.weight + layer.bias
                        elif 'GELU' in layer._get_name() or 'Softmax' in layer._get_name():
                            x_ = x
                        else:
                            x_ = layer.old_forward(x)

                        # Filter specific layers
                        if any(n in layer._get_name() for n in ['Conv', 'BatchNorm', 'LayerNorm', 'Embedding']):
                            x_ = filter(x_, dim=dim, target_shape=target_shape, split=split)

                        return x_

                    return lambda_forward

                child.old_forward = child.forward
                local_shape = local_child.weight.shape if hasattr(local_child, 'weight') else None
                child.forward = new_forward(child, local_shape)
            else:
                modify_forward(child, local_child, split)

    model_kwargs = copy.deepcopy(model.stored_inp_kwargs)

    # Disable dropout for consistency
    if 'cfg' in model_kwargs:
        model_kwargs['cfg']['dropout_rate'] = 0
        model_kwargs['cfg']['drop_connect_rate'] = 0
    if 'config' in model_kwargs:
        for key in [k for k in model_kwargs['config'].__dict__ if 'dropout' in k]:
            setattr(model_kwargs['config'], key, 0)

    copy_model = type(model)(**model_kwargs)
    if args.use_gpu:
        copy_model = copy_model.cuda()

    if 'scale' in model_kwargs:
        model_kwargs['scale'] = scale
    else:
        model_kwargs['params']['scale'] = scale

    local_model = type(model)(**model_kwargs)

    modify_forward(copy_model, local_model)

    # Initialize model state with uniform values
    state_dict = copy_model.state_dict(keep_vars=True)
    for k, v in state_dict.items():
        if 'num_batches_tracked' not in k:
            state_dict[k] = torch.ones_like(v) / math.sqrt(v.numel())

    copy_model.load_state_dict(state_dict)

    # Perform a dummy backward pass
    loss_obj = nn.MSELoss()
    inp = torch.ones((1, 3, args.image_size[0], args.image_size[1]))
    if args.use_gpu:
        inp = inp.cuda()
    preds = copy_model(inp)[0]
    target = torch.ones_like(preds[-1])
    if args.use_gpu:
        target = target.cuda()
    loss = sum(loss_obj(pred, target) for pred in preds)
    loss.backward()

    # Create masks for parameter retention
    state_dict = copy_model.state_dict(keep_vars=True)
    idx_dict = {}
    for k, v in state_dict.items():
        if 'num_batches_tracked' not in k:
            idx_dict[k] = v.grad != 0 if v.grad is not None else torch.ones_like(v, dtype=bool)

    restore_forward(copy_model)

    return idx_dict
