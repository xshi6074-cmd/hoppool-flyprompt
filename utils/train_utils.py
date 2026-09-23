# import torch_optimizer
# from easydict import EasyDict as edict
from torch import optim

from models import MODELS
from models.backbone import create_backbone
from optim.fam import FAM
from optim.sam import SAM


# Methods whose model state advances on internal sample-count steps
# (``--step_num``) instead of benchmark task ids.
STEP_AWARE_METHODS = frozenset({
    "dualprompt", "mvp", "flyprompt", "mlp_generator",
    "baseline", "shared_prompt", "hfpool", "gate",
})


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


def _split_hopfield_slice_masked(parameters):
    regular_params = []
    slice_masked_params = []
    for param in parameters:
        if not param.requires_grad:
            continue
        if getattr(param, "_hopfield_slice_masked", False):
            slice_masked_params.append(param)
        else:
            regular_params.append(param)
    return regular_params, slice_masked_params


def _sgd_parameter_groups(parameters, weight_decay):
    regular_params, slice_masked_params = _split_hopfield_slice_masked(
        parameters
    )
    groups = [{"params": regular_params, "weight_decay": weight_decay}]
    if slice_masked_params:
        groups.append({"params": slice_masked_params, "weight_decay": 0.0})
    return groups


def select_optimizer(opt_name, lr, model):

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    if opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif opt_name == 'adam_adapt':
        fc_params = []
        other_params = []
        fc_params_name = []
        other_params_name = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'fc.' in name:  # If the parameter is from a fully-connected layer
                    fc_params.append(param)
                    fc_params_name.append(name)
                else:  # All other layers
                    other_params.append(param)
                    other_params_name.append(name)
        opt = optim.Adam([
                        {'params': fc_params, 'lr': lr},       # Learning rate lr1 for fully-connected layers
                        {'params': other_params, 'lr': lr*5}     # Learning rate lr2 for all other layers
                    ], weight_decay=0)
    elif opt_name == "sgd":
        opt = optim.SGD(
            _sgd_parameter_groups(model.parameters(), weight_decay=1e-4),
            lr=lr,
            momentum=0.9,
            nesterov=True,
        )
    elif opt_name == 'sgd_sl':
        fc_params = []
        other_params = []
        fc_params_name = []
        other_params_name = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'fc.' in name:  # If the parameter is from a fully-connected layer
                    fc_params.append(param)
                    fc_params_name.append(name)
                else:  # All other layers
                    other_params.append(param)
                    other_params_name.append(name)
        other_params, slice_masked_params = _split_hopfield_slice_masked(
            other_params
        )
        parameter_groups = [
            {'params': other_params, 'lr': lr, 'weight_decay': 5e-4},
            {'params': fc_params, 'lr': 0.005, 'weight_decay': 5e-4},
        ]
        if slice_masked_params:
            parameter_groups.append({
                'params': slice_masked_params,
                'lr': lr,
                'weight_decay': 0.0,
            })
        opt = optim.SGD(parameter_groups)
    elif opt_name == "sam":
        base_optimizer = optim.Adam
        opt = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=0)
    elif opt_name == "fam":
        base_optimizer = optim.Adam
        opt = FAM(model.parameters(), base_optimizer, lr=lr, weight_decay=0)
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    return opt

def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2)
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80, 90], gamma=0.1)
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    elif sched_name == "sam":
        scheduler = optim.lr_scheduler.LambdaLR(opt.base_optimizer, lambda iter: 1)
    elif sched_name == "fam":
        scheduler = optim.lr_scheduler.LambdaLR(opt.base_optimizer, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler

def select_model(method, backbone, num_classes=None, n_tasks=None, kwargs=None):
    if method=="slca":
        model = create_backbone(
            backbone,
            pretrained=True,
            num_classes=num_classes,
            backbone_path=(kwargs or {}).get("backbone_path"),
            drop_rate=0.,
            drop_path_rate=0.,
            drop_block_rate=None,
        )
    elif method in MODELS.keys():
        # For most methods, task_num corresponds to the benchmark number of
        # tasks (n_tasks). For some prompt-based methods (DualPrompt, MVP,
        # FlyPrompt), we instead interpret task_num as the number of internal
        # steps, which can be overridden by ``step_num`` if provided.
        task_num_for_model = n_tasks
        if kwargs is not None and method in STEP_AWARE_METHODS:
            step_num = kwargs.get("step_num", None)
            if step_num is not None and step_num > 0:
                task_num_for_model = step_num

        model = MODELS[method](
            backbone_name=backbone,
            pretrained=True,
            num_classes=num_classes,
            task_num=task_num_for_model,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Unsupported method: {method}")

    return model
