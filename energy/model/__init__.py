

def check_gradient_norm_L2(model):
    total_norm = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm

    return total_norm