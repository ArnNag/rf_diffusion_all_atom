import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
import copy
import dgl
from util import base_indices, RTs_by_torsion, xyzs_in_base_frame, rigid_from_3_points


def make_rotX(angs, eps=1e-6):
    B, L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4, device=angs.device).repeat(B, L, 1, 1)

    RTs[:, :, 1, 1] = angs[:, :, 0] / NORM
    RTs[:, :, 1, 2] = -angs[:, :, 1] / NORM
    RTs[:, :, 2, 1] = angs[:, :, 1] / NORM
    RTs[:, :, 2, 2] = angs[:, :, 0] / NORM
    return RTs


# rotate about the z axis
def make_rotZ(angs, eps=1e-6):
    B, L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4, device=angs.device).repeat(B, L, 1, 1)

    RTs[:, :, 0, 0] = angs[:, :, 0] / NORM
    RTs[:, :, 0, 1] = -angs[:, :, 1] / NORM
    RTs[:, :, 1, 0] = angs[:, :, 1] / NORM
    RTs[:, :, 1, 1] = angs[:, :, 0] / NORM
    return RTs


# rotate about an arbitrary axis
def make_rot_axis(angs, u, eps=1e-6):
    B, L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4, device=angs.device).repeat(B, L, 1, 1)

    ct = angs[:, :, 0] / NORM
    st = angs[:, :, 1] / NORM
    u0 = u[:, :, 0]
    u1 = u[:, :, 1]
    u2 = u[:, :, 2]

    RTs[:, :, 0, 0] = ct + u0 * u0 * (1 - ct)
    RTs[:, :, 0, 1] = u0 * u1 * (1 - ct) - u2 * st
    RTs[:, :, 0, 2] = u0 * u2 * (1 - ct) + u1 * st
    RTs[:, :, 1, 0] = u0 * u1 * (1 - ct) + u2 * st
    RTs[:, :, 1, 1] = ct + u1 * u1 * (1 - ct)
    RTs[:, :, 1, 2] = u1 * u2 * (1 - ct) - u0 * st
    RTs[:, :, 2, 0] = u0 * u2 * (1 - ct) - u1 * st
    RTs[:, :, 2, 1] = u1 * u2 * (1 - ct) + u0 * st
    RTs[:, :, 2, 2] = ct + u2 * u2 * (1 - ct)
    return RTs


class ComputeAllAtomCoords(nn.Module):
    def __init__(self):
        super(ComputeAllAtomCoords, self).__init__()

        self.base_indices = nn.Parameter(base_indices, requires_grad=False)
        self.RTs_in_base_frame = nn.Parameter(RTs_by_torsion, requires_grad=False)
        self.xyzs_in_base_frame = nn.Parameter(xyzs_in_base_frame, requires_grad=False)

    def forward(self, seq, xyz, alphas, non_ideal=False) -> tuple[Tensor, Tensor]:
        B, L = xyz.shape[:2]

        Rs, Ts = rigid_from_3_points(N=xyz[..., 0, :], Ca=xyz[..., 1, :], C=xyz[..., 2, :], non_ideal=non_ideal)

        RTF0 = torch.eye(4).repeat(B, L, 1, 1).to(device=Rs.device)
        # output shape: (B, L, 4, 4)
        # TODO: what are the two 4 axes? The first 3 elements of the third axis are being set from Rs and Ts. What's the
        # fourth element for?

        # bb
        RTF0[:, :, :3, :3] = Rs
        RTF0[:, :, :3, 3] = Ts

        # TODO: what do the letters used in einsum mean? b = batch, r = length of sequence
        # omega
        RTF1 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq, 0, :], make_rotX(alphas[:, :, 0, :]))

        # phi
        RTF2 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq, 1, :], make_rotX(alphas[:, :, 1, :]))

        # psi
        RTF3 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq, 2, :], make_rotX(alphas[:, :, 2, :]))

        # CB bend
        basexyzs = self.xyzs_in_base_frame[seq]
        NCr = 0.5 * (basexyzs[:, :, 2, :3] + basexyzs[:, :, 0, :3])
        CAr = (basexyzs[:, :, 1, :3])
        CBr = (basexyzs[:, :, 4, :3])
        CBrotaxis1 = (CBr - CAr).cross(NCr - CAr)
        CBrotaxis1 /= torch.linalg.norm(CBrotaxis1, dim=-1, keepdim=True) + 1e-8

        # CB twist
        NCp = basexyzs[:, :, 2, :3] - basexyzs[:, :, 0, :3]
        NCpp = NCp - torch.sum(NCp * NCr, dim=-1, keepdim=True) / torch.sum(NCr * NCr, dim=-1, keepdim=True) * NCr
        CBrotaxis2 = (CBr - CAr).cross(NCpp)
        CBrotaxis2 /= torch.linalg.norm(CBrotaxis2, dim=-1, keepdim=True) + 1e-8

        CBrot1 = make_rot_axis(alphas[:, :, 7, :], CBrotaxis1)
        CBrot2 = make_rot_axis(alphas[:, :, 8, :], CBrotaxis2)

        RTF8 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, CBrot1, CBrot2)

        # chi1 + CG bend
        RTF4 = torch.einsum(
            'brij,brjk,brkl,brlm->brim',
            RTF8,
            self.RTs_in_base_frame[seq, 3, :],
            make_rotX(alphas[:, :, 3, :]),
            make_rotZ(alphas[:, :, 9, :]))

        # chi2
        RTF5 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF4, self.RTs_in_base_frame[seq, 4, :], make_rotX(alphas[:, :, 4, :]))

        # chi3
        RTF6 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF5, self.RTs_in_base_frame[seq, 5, :], make_rotX(alphas[:, :, 5, :]))

        # chi4
        RTF7 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF6, self.RTs_in_base_frame[seq, 6, :], make_rotX(alphas[:, :, 6, :]))

        RTframes = torch.stack((
            RTF0, RTF1, RTF2, RTF3, RTF4, RTF5, RTF6, RTF7, RTF8
        ), dim=2)

        xyzs = torch.einsum(
            'brtij,brtj->brti',
            RTframes.gather(2, self.base_indices[seq][..., None, None].repeat(1, 1, 1, 4, 4)), basexyzs
        )

        # TODO: this was marked as use_H. Why 3? Is this selecting the backbone frame atoms?
        return RTframes, xyzs[..., :3]
