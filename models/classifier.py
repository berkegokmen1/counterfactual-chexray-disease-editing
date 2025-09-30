from torchvision.models import densenet121, resnet152
from chexpert.models.efficientnet import construct_model
from chexpert.models.attn_aug_conv import DenseNet, ResNet, Bottleneck
from chexpert.dataset import ChexpertSmall, extract_patient_ids
from torchvision.models.densenet import DenseNet121_Weights

import torch
import torch.nn as nn
import numpy as np
import os


def load_model(args):

    classifier_args = args.model.classifier
    model_name, restore, restore_path, lr, pretrained = (
        classifier_args.model.strip().lower(),
        classifier_args.restore,
        classifier_args.path,
        classifier_args.lr,
        classifier_args.pretrained,
    )

    print("Loading model", model_name)

    # load model
    n_classes = len(ChexpertSmall.attr_names)
    if model_name == "densenet121":
        model = densenet121(weights=(DenseNet121_Weights.DEFAULT if pretrained else None)).to(args.device)
        # 1. replace output layer with chexpert number of classes (pretrained loads ImageNet n_classes)
        model.classifier = nn.Linear(model.classifier.in_features, out_features=n_classes).to(args.device)
        # 2. init output layer with default torchvision init
        nn.init.constant_(model.classifier.bias, 0)
        # 3. store locations of forward and backward hooks for grad-cam
        grad_cam_hooks = {"forward": model.features.norm5, "backward": model.classifier}
        # 4. init optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
    #        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    #        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40000, 60000])
    elif model_name == "aadensenet121":
        model = DenseNet(
            32,
            (6, 12, 24, 16),
            64,
            num_classes=n_classes,
            attn_params={"k": 0.2, "v": 0.1, "nh": 8, "relative": True, "input_dims": (320, 320)},
        ).to(args.device)
        grad_cam_hooks = {"forward": model.features, "backward": model.classifier}
        attn_hooks = [model.features.transition1.conv, model.features.transition2.conv, model.features.transition3.conv]
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40000, 60000])
    elif model_name == "resnet152":
        model = resnet152(weights=pretrained).to(args.device)
        model.fc = nn.Linear(model.fc.in_features, out_features=n_classes).to(args.device)
        grad_cam_hooks = {"forward": model.layer4, "backward": model.fc}
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
    elif (
        model_name == "aaresnet152"
    ):  # resnet50 layers [3,4,6,3]; resnet101 layers [3,4,23,3]; resnet 152 layers [3,8,36,3]
        model = ResNet(
            Bottleneck,
            [3, 8, 36, 3],
            num_classes=n_classes,
            attn_params={"k": 0.2, "v": 0.1, "nh": 8, "relative": True, "input_dims": (320, 320)},
        ).to(args.device)
        grad_cam_hooks = {"forward": model.layer4, "backward": model.fc}
        attn_hooks = (
            [model.layer2[i].conv2 for i in range(len(model.layer2))]
            + [model.layer3[i].conv2 for i in range(len(model.layer3))]
            + [model.layer4[i].conv2 for i in range(len(model.layer4))]
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
    elif "efficientnet" in model_name:
        model = construct_model(model_name, n_classes=n_classes).to(args.device)
        grad_cam_hooks = {"forward": model.head[1], "backward": model.head[-1]}
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, eps=0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay_factor)
    else:
        raise RuntimeError("Model architecture not supported.")

    if restore and os.path.isfile(restore_path):
        print("Restoring model weights from {}".format(restore_path))
        model_checkpoint = torch.load(restore_path, map_location=args.device, weights_only=False)
        model.load_state_dict(model_checkpoint["state_dict"])
        args.model.classifier.step = model_checkpoint["global_step"]
        del model_checkpoint

        model = model.to(args.device)

    return model, grad_cam_hooks


def define_classifier_model(args):
    model, _ = load_model(args)
    model = model.to(args.device)

    model.eval()

    classifier_name = args.model.classifier.model.lower()

    print("Using classifier:", classifier_name)

    return model
