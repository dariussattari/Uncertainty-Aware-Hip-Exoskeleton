"""A thigh IMU worn at a different fit: rotated about a random axis and moved along / across the thigh (Model B1).

Both IMUs sit on one rigid segment, so for an IMU displaced by r (m, in the Nature thigh IMU's frame) and then rotated
by R:

    gyro'  = R gyro                                        (angular velocity is the same everywhere on a rigid body)
    accel' = R (accel + alpha x r + omega x (omega x r))   (tangential and centripetal terms; gravity is common to both)

with omega in rad/s and alpha = d omega / dt. The Nature thigh frame is x anterior, y along the thigh (up), z lateral
(nature notebook 01, section 3.4; ``nature_data.NATURE_ANATOMICAL``). Units as published: accel m/s^2, gyro deg/s.
"""
import math

import torch


def get_rotation(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: (B, 3) axis * angle (rad) -> (B, 3, 3) rotation matrices."""
    angle = axis_angle.norm(dim = 1, keepdim = True).clamp(min = 1e-12)       # (B, 1)
    k = axis_angle / angle                                                     # (B, 3) unit axis
    zero = torch.zeros_like(k[:, 0])
    cross = torch.stack([torch.stack([zero, -k[:, 2], k[:, 1]], 1), torch.stack([k[:, 2], zero, -k[:, 0]], 1),
                         torch.stack([-k[:, 1], k[:, 0], zero], 1)], 1)       # (B, 3, 3) [k]x
    eye = torch.eye(3, device = k.device, dtype = k.dtype).expand_as(cross)
    s, c = torch.sin(angle)[:, :, None], torch.cos(angle)[:, :, None]
    return eye + s * cross + (1 - c) * cross @ cross


def sample_fits(n: int, max_rotation_deg: float, lever_along_m: float, lever_across_m: float, generator: torch.Generator,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """n random fits: rotations (n, 3, 3) about a uniform random axis by an angle uniform in [0, max], and lever arms
    (n, 3) uniform in +-across (x, z) and +-along (y)."""
    axis = torch.randn(n, 3, generator = generator)
    axis = axis / axis.norm(dim = 1, keepdim = True)
    angle = torch.rand(n, 1, generator = generator) * math.radians(max_rotation_deg)
    bound = torch.tensor([lever_across_m, lever_along_m, lever_across_m])
    lever = (2 * torch.rand(n, 3, generator = generator) - 1) * bound
    return get_rotation(axis * angle).to(device), lever.to(device)


def perturb_imu(accel: torch.Tensor, gyro: torch.Tensor, rotation: torch.Tensor, lever: torch.Tensor, rate_hz: float
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """The same motion seen by the IMU at each window's fit. accel, gyro (B, T, 3); rotation (B, 3, 3); lever (B, 3)."""
    omega = torch.deg2rad(gyro)                                                # (B, T, 3) rad/s
    alpha = torch.gradient(omega, spacing = 1.0 / rate_hz, dim = 1)[0]         # (B, T, 3) rad/s^2
    r = lever[:, None, :].expand_as(omega)                                     # (B, 3) -> (B, T, 3)
    moved = accel + torch.linalg.cross(alpha, r) + torch.linalg.cross(omega, torch.linalg.cross(omega, r))
    return torch.einsum('bij,btj->bti', rotation, moved), torch.einsum('bij,btj->bti', rotation, gyro)
