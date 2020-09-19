import torch
import torch.nn.functional as F


def DiffAugment(x, policy='', channels_first=True):
    # why need to permute
    # https://github.com/ninetf135246/pytorch-Learning-to-See-in-the-Dark/issues/1#issuecomment-434601063
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def augment_brightness(x):
    # In paper, it said that random brightness is within [−0.5, 0.5],
    rand_number = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5
    # original image plus the jitter brightness and get the bright image
    x += rand_number
    return x


def augment_saturation(x):
    # dim 1 is the image channel, so from dim 1 do mean operation, compute the mean value from RGB
    # (or single channel) of each pixel
    x_mean = x.mean(dim=1, keepdim=True)
    # In paper, it said that saturation is within [0, 2]
    rand_number = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    # original image plus the jitter saturation and get the result
    x = (x - x_mean) * rand_number + x_mean
    return x


def augment_contrast(x):
    # compute the whole image's average value
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    # In paper, it said that contrast is within [0.5, 1.5]
    rand_number = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    # original image plus the jitter contrast and get the result
    x = (x - x_mean) * rand_number + x_mean
    return x


def augment_translation(x, ratio=0.125):
    batch = x.size(0)
    channel = x.size(1)
    h = x.size(2)
    w = x.size(3)

    # In paper, it said that translation is within [−1/8, 1/8] of the image size,
    shifting_x, shifting_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    # shift horizontally and vertically
    translation_x = torch.randint(-shifting_x, shifting_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shifting_y, shifting_y + 1, size=[x.size(0), 1, 1], device=x.device)
    # get the grid of the x, get the output would also have k tensors
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    # Clamp all elements in input into the range [ 0, size] and return a resulting tensor
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    # padded with zeros
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    #
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def augment_cutout(x, ratio=0.5):
    batch = x.size(0)
    channel = x.size(1)
    h = x.size(2)
    w = x.size(3)

    # In paper, it's said that cutout is masking with a random square of half image size
    cutout_size = int(h * ratio + 0.5), int(w * ratio + 0.5)
    # compute the offset of x and y 
    offset_x = torch.randint(0, h + (1 - cutout_size[0] % 2), size=[batch, 1, 1], device=x.device)
    offset_y = torch.randint(0, w + (1 - cutout_size[1] % 2), size=[batch, 1, 1], device=x.device)
    # get the grid of the x, get the output would also have k tensors
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(batch, dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    # Clamp all elements in input into the range [ 0, size] and return a resulting tensor
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, 0, h - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, 0, w - 1)
    #generate original mask with the size of x
    mask = torch.ones(batch, h, w, dtype=x.dtype, device=x.device)
    #for the cutout pixels, set to 0
    mask[grid_batch, grid_x, grid_y] = 0
    # x times mask, so the x's pixels's value will come to 0 if mask's pixels come to 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [augment_brightness, augment_saturation, augment_contrast],
    'translation': [augment_translation],
    'cutout': [augment_cutout],
}
