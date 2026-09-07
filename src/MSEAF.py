import torch
import torch.nn as nn
import torch.nn.functional as F


class SSM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Sau MaxPool + MeanPool theo channel:
        # [B, C, H, W] -> [B, 2, H, W]
        # Conv 3x3 -> [B, 1, H, W]
        self.spatial_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        # Nhánh dưới:
        # Depth-wise Conv 5x5, dilation=2
        # -> Depth-wise Conv 7x7, dilation=3
        self.dw_5_7 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                stride=1,
                padding=4,
                dilation=2,
                groups=channels,
                bias=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=7,
                stride=1,
                padding=9,
                dilation=3,
                groups=channels,
                bias=True
            )
        )

        # Nhánh trên:
        # Depth-wise Conv 3x3
        self.dw_3 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=True
        )

    def forward(self, x):

        # --------------------------------
        # 1. Pool theo chiều channel
        # --------------------------------

        max_pool = torch.max(
            x, dim=1, keepdim=True
        )[0]

        mean_pool = torch.mean(
            x, dim=1, keepdim=True
        )

        # max_pool  : [B, 1, H, W]
        # mean_pool : [B, 1, H, W]

        pooled = torch.cat(
            [max_pool, mean_pool],
            dim=1
        )

        # pooled: [B, 2, H, W]

        # --------------------------------
        # 2. Conv 3x3 tạo spatial map F'
        # --------------------------------

        spatial_map = self.spatial_conv(pooled)

        # spatial_map: [B, 1, H, W]

        # --------------------------------
        # 3. Hai nhánh Depth-wise Conv
        # --------------------------------

        branch_5_7 = self.dw_5_7(x)
        branch_3 = self.dw_3(x)

        # cả hai:
        # [B, C, H, W]

        # --------------------------------
        # 4. Multiply + Add
        # --------------------------------

        out = branch_5_7 * spatial_map + branch_3

        return out
class FSM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Hai hệ số học được
        # Official code:
        # a khởi tạo = 0
        # b khởi tạo = 1
        self.a = nn.Parameter(torch.zeros(channels, 1, 1))
        self.b = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x):
        # x: [B, C, H, W]

        # 1. Mean theo H, W
        mean = torch.mean(
            x,
            dim=(2, 3),
            keepdim=True
        )
        # mean: [B, C, 1, 1]

        # 2. Lấy high-frequency component
        high = x - mean
        # high: [B, C, H, W]

        # 3. Reweight high-frequency rồi residual
        out = self.a * high * x + self.b * x

        return out
    
class DSM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.ssm = SSM(channels)
        self.fsm = FSM(channels)

    def forward(self, x):
        # x: [B, C, H, W]

        x = self.ssm(x)
        # sau SSM vẫn: [B, C, H, W]

        x = self.fsm(x)
        # sau FSM vẫn: [B, C, H, W]

        return x
class EdgeEnhancer(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # PAPER KHÔNG ghi rõ kernel của AvgPool.
        # Tạm chọn 3x3, stride=1, padding=1 để giữ nguyên H,W.
        self.avg_pool = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1
        )

        # PAPER chỉ nói "convolutional layers",
        # không ghi rõ kernel / BN / activation.
        # Tạm dùng Conv 3x3 giữ nguyên channel.
        self.edge_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        # x: [B, C, H, W]

        # 1. Low-frequency / smoothed feature
        smooth = self.avg_pool(x)

        # 2. High-frequency / edge
        edge = x - smooth

        # 3. Transform edge feature
        edge = self.edge_conv(edge)

        # 4. Residual connection
        out = x + edge

        return out
class MSEAFBranch(nn.Module):
    def __init__(self, channels, pool_size):
        super().__init__()

        reduced_channels = channels // 4

        # Eq. (1)
        # [B, C, H, W] -> [B, C, s, s]
        self.pool = nn.AdaptiveAvgPool2d(
            output_size=(pool_size, pool_size)
        )

        # Eq. (2)
        # C -> C/4
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=reduced_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Feature extraction
        # C/4 -> C/4
        self.conv3 = nn.Conv2d(
            in_channels=reduced_channels,
            out_channels=reduced_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Edge enhancement
        self.edge = EdgeEnhancer(reduced_channels)

    def forward(self, x):
        # x: [B, C, H, W]

        # Lưu H, W ban đầu để lát upsample lại
        h, w = x.shape[-2:]

        # 1. AdaptivePool
        x = self.pool(x)

        # 2. Conv 1x1: C -> C/4
        x = self.conv1(x)

        # 3. Conv 3x3
        x = self.conv3(x)

        # 4. EdgeEnhancer
        x = self.edge(x)

        # 5. Upsample về H, W ban đầu
        x = F.interpolate(
            x,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        return x
class MSEAF(nn.Module):
    def __init__(self, channels):
        super().__init__()

        assert channels % 4 == 0, "channels phải chia hết cho 4"

        # 4 nhánh multi-scale
        self.branch3 = MSEAFBranch(
            channels=channels,
            pool_size=3
        )

        self.branch6 = MSEAFBranch(
            channels=channels,
            pool_size=6
        )

        self.branch9 = MSEAFBranch(
            channels=channels,
            pool_size=9
        )

        self.branch12 = MSEAFBranch(
            channels=channels,
            pool_size=12
        )

        # Nhánh thứ 5:
        # giữ original-resolution feature

        self.original_branch = nn.Conv2d( in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1 )
        """
        self.original_branch = nn.Sequential(
            # Depthwise 3x3
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels
            ),

            # Pointwise 1x1
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        """
        # Sau concat:
        # 4*(C/4) + C = 2C
        self.dsm = DSM(
            channels=2 * channels
        )

        # Eq. (5):
        # 2C -> C
        self.final_conv = nn.Conv2d(
            in_channels=2 * channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):

        # -----------------------------------
        # 1. 4 multi-scale branches
        # -----------------------------------

        f1 = self.branch3(x)
        f2 = self.branch6(x)
        f3 = self.branch9(x)
        f4 = self.branch12(x)

        # -----------------------------------
        # 2. Original branch
        # -----------------------------------

        f5 = self.original_branch(x)

        # -----------------------------------
        # 3. Concat
        # -----------------------------------

        x = torch.cat(
            [f1, f2, f3, f4, f5],
            dim=1
        )

        # x: [B, 2C, H, W]

        # -----------------------------------
        # 4. DSM
        # -----------------------------------

        x = self.dsm(x)

        # -----------------------------------
        # 5. Final 1x1 Conv
        # 2C -> C
        # -----------------------------------

        x = self.final_conv(x)

        return x