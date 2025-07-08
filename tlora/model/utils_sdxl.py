import torch


def count_trainable_params(module, verbose=False):
    numel = 0
    for name, param in module.named_parameters():
        if param.requires_grad:
            numel += torch.numel(param)
            if verbose:
                print(name, torch.numel(param), sep='\t\t')

    return numel


@torch.no_grad()
def params_grad_norm(parameters):
    grad_norm_squared = 0
    for param in parameters:
        if param.grad is not None:
            grad_norm_squared += torch.square(torch.linalg.norm(param.grad)).item()

    return grad_norm_squared ** 0.5


def cast_training_params(models, dtype=torch.float32):
    """
    Casts the training parameters of the model to the specified data type.

    Args:
        models: The PyTorch model whose parameters will be cast.
        dtype: The data type to which the model parameters will be cast.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    for m in models:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)
