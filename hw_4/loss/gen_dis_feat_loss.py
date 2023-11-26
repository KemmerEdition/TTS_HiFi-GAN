import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return 2 * loss


def discriminator_loss(real_out, gen_out):
    loss = 0
    for dr, dg in zip(real_out, gen_out):
        r_loss = torch.mean((1 - dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
    return loss


def generator_loss(d_outs):
    loss = 0
    for dg in d_outs:
        l = torch.mean((1 - dg)**2)
        loss += l
    return loss
