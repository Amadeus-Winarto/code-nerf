import torch
import torch.nn as nn


def PE(x, degree):
    y = torch.cat([2.0**i * x for i in range(degree)], -1)
    w = 1
    return torch.cat([x] + [torch.sin(y) * w, torch.cos(y) * w], -1)


class CodeNeRF(nn.Module):
    def __init__(
        self,
        shape_blocks=2,
        texture_blocks=1,
        W=256,
        num_xyz_freq=10,
        num_dir_freq=4,
        latent_dim=256,
    ):
        super().__init__()
        self.shape_blocks = shape_blocks
        self.texture_blocks = texture_blocks
        self.num_xyz_freq = num_xyz_freq
        self.num_dir_freq = num_dir_freq

        d_xyz, d_viewdir = 3 + 6 * num_xyz_freq, 3 + 6 * num_dir_freq
        self.encoding_xyz = nn.Sequential(nn.Linear(d_xyz, W), nn.ReLU())
        for j in range(shape_blocks):
            layer = nn.Sequential(nn.Linear(latent_dim, W), nn.ReLU())
            setattr(self, f"shape_latent_layer_{j+1}", layer)
            layer = nn.Sequential(nn.Linear(W, W), nn.ReLU())
            setattr(self, f"shape_layer_{j+1}", layer)
        self.encoding_shape = nn.Linear(W, W)
        self.sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.encoding_viewdir = nn.Sequential(nn.Linear(W + d_viewdir, W), nn.ReLU())
        for j in range(texture_blocks):
            layer = nn.Sequential(nn.Linear(latent_dim, W), nn.ReLU())
            setattr(self, f"texture_latent_layer_{j+1}", layer)
            layer = nn.Sequential(nn.Linear(W, W), nn.ReLU())
            setattr(self, f"texture_layer_{j+1}", layer)
        self.rgb = nn.Sequential(nn.Linear(W, W // 2), nn.ReLU(), nn.Linear(W // 2, 3))

    def forward(self, xyz, viewdir, shape_latent, texture_latent):
        xyz = PE(xyz, self.num_xyz_freq)
        viewdir = PE(viewdir, self.num_dir_freq)
        y = self.encoding_xyz(xyz)
        for j in range(self.shape_blocks):
            z = getattr(self, f"shape_latent_layer_{j+1}")(shape_latent)
            y = y + z
            y = getattr(self, f"shape_layer_{j+1}")(y)
        y = self.encoding_shape(y)
        sigmas = self.sigma(y)
        y = torch.cat([y, viewdir], -1)
        y = self.encoding_viewdir(y)
        for j in range(self.texture_blocks):
            z = getattr(self, f"texture_latent_layer_{j+1}")(texture_latent)
            y = y + z
            y = getattr(self, f"texture_layer_{j+1}")(y)
        rgbs = self.rgb(y)
        return sigmas, rgbs


class TriplaneCodeNeRF(CodeNeRF):
    def __init__(
        self,
        shape_blocks=2,
        texture_blocks=1,
        W=256,
        num_xyz_freq=10,
        num_dir_freq=4,
        latent_dim=256,
    ):
        super().__init__(
            shape_blocks, texture_blocks, W, num_xyz_freq, num_dir_freq, 3 * latent_dim
        )  # 3 * latent_dim because we have 3 planes

    def sample_triplane(self, xyz, latents):
        # xyz: (num_rays, num_points, 3)
        # latents: (1, 3 * latent_dim * 128 * 128)

        num_rays = xyz.shape[0]
        num_points = xyz.shape[1]

        # Form triplanes
        latents = latents.view(latents.shape[0], 3, -1, 128, 128)
        latent_xy = latents[:, 0]  # (1, latent_dim, 128, 128)
        latent_xz = latents[:, 1]
        latent_yz = latents[:, 2]

        # Sample the latent codes from each plane
        xyz = xyz.unsqueeze(0).reshape(1, -1, 3)  # (1, num_rays * num_points, 3)
        xy = xyz[:, :, :2].unsqueeze(1)  # (1, 1, num_rays * num_points, 2)
        xz = xyz[:, :, [0, 2]].unsqueeze(1)
        yz = xyz[:, :, 1:].unsqueeze(1)

        latent_xy = torch.nn.functional.grid_sample(latent_xy, xy, align_corners=True)
        latent_xz = torch.nn.functional.grid_sample(latent_xz, xz, align_corners=True)
        latent_yz = torch.nn.functional.grid_sample(latent_yz, yz, align_corners=True)

        # Concatenate the latent codes
        latent = torch.cat([latent_xy, latent_xz, latent_yz], -2)

        # (num_rays, num_scenes, num_points, 3 * latent_dim)
        latent = latent.permute(3, 0, 2, 1).reshape(
            num_rays, num_points, -1
        )  # (num_rays, num_points, 3 * latent_dim)
        return latent

    def forward(self, xyz, viewdir, shape_latent, texture_latent):
        # Sample shape latents
        shape_latent = self.sample_triplane(xyz, shape_latent)  # (B, N, 3 * latent_dim)
        texture_latent = self.sample_triplane(xyz, texture_latent)
        return super().forward(xyz, viewdir, shape_latent, texture_latent)


if __name__ == "__main__":
    num_rays = 2048
    num_points = 96
    latent_dim = 6
    model = TriplaneCodeNeRF(latent_dim=latent_dim)

    xyz = torch.rand(num_rays, num_points, 3)
    viewdir = torch.rand(num_rays, num_points, 3)
    shape_latent = torch.rand(1, 3 * latent_dim * 128 * 128)
    texture_latent = torch.rand(1, 3 * latent_dim * 128 * 128)

    sigmas, rgbs = model(xyz, viewdir, shape_latent, texture_latent)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    original_model = CodeNeRF(latent_dim=256)
    print("Model: ", count_parameters(model))
    print("Original: ", count_parameters(original_model))
