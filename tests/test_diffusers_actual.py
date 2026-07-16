import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from stitchdiff.model import configure_unet_memory, expand_unet_conv_in


def test_real_diffusers_unet_fourteen_channel_forward_backward():
    unet = diffusers.UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32,),
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",),
        cross_attention_dim=16,
        attention_head_dim=8,
        norm_num_groups=8,
    )
    old_weight = unet.conv_in.weight.detach().clone()
    expand_unet_conv_in(unet, 14)
    configure_unet_memory(
        unet,
        gradient_checkpointing=True,
        channels_last=True,
    )
    sample = torch.randn(1, 14, 16, 16)
    timestep = torch.tensor([5])
    hidden = torch.zeros(1, 5, 16)
    prediction = unet(sample, timestep, encoder_hidden_states=hidden).sample
    prediction.square().mean().backward()

    assert prediction.shape == (1, 4, 16, 16)
    torch.testing.assert_close(unet.conv_in.weight[:, :4], old_weight)
    torch.testing.assert_close(
        unet.conv_in.weight[:, 4:], torch.zeros_like(unet.conv_in.weight[:, 4:])
    )
    assert unet.conv_in.weight.grad is not None
