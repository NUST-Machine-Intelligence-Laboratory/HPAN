from torch.optim import Adam


def PANNet_optimizer(model, lr=5e-5, betas=(0.9, 0.999), weight_decay=5e-4):
    for param in model.encoder.parameters():
        param.requires_grad = False

    opt = Adam([{
        'params': model.proto_net.parameters()
    }, {
        'params': model.domain_net.parameters()
    }, {
        'params': model.decoder.parameters()
    }],
               lr=lr,
               betas=betas,
               weight_decay=weight_decay)
    return opt


def finetune_optimizer(model, lr=2e-5, betas=(0.9, 0.999), weight_decay=5e-4):
    for param in model.encoder.layer0.parameters():
        param.requires_grad = False
    for param in model.encoder.layer1.parameters():
        param.requires_grad = False
    for param in model.encoder.layer2.parameters():
        param.requires_grad = False

    opt = Adam([{
        'params': model.encoder.layer3.parameters()
    }, {
        'params': model.encoder.layer4.parameters()
    }],
               lr=lr,
               betas=betas,
               weight_decay=weight_decay)
    return opt
