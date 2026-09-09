import pytest

torch = pytest.importorskip("torch")

from lenscraft.models import AAE, DCAE, VAE, build_anomaly_model, build_classifier, conv_feature_size  # noqa: E402


def test_paper_feature_size_at_150():
    assert conv_feature_size(150) == 5184  # 64 x 9 x 9, Table IV


@pytest.mark.parametrize("arch", ["resnet18", "alexnet"])
def test_classifier_shapes(arch):
    model = build_classifier(arch).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 1, 150, 150))
    assert out.shape == (2, 3)


@pytest.mark.parametrize("size", [96, 150, 204])
@pytest.mark.parametrize("arch", ["dcae", "vae", "aae"])
def test_autoencoders_reconstruct_exact_size(arch, size):
    model = build_anomaly_model(arch, latent_dim=16, input_size=size).eval()
    x = torch.rand(2, 1, size, size)
    with torch.no_grad():
        recon = model.reconstruct(x)
        out = model(x)
    assert recon.shape == x.shape
    assert recon.min() >= -1 and recon.max() <= 1  # tanh output
    if arch == "dcae":
        assert out.shape == x.shape
    elif arch == "vae":
        assert out[0].shape == x.shape and out[1].shape == (2, 16) and out[2].shape == (2, 16)
    else:
        assert out[0].shape == x.shape and out[1].shape == (2, 16)
        assert model.discriminator(out[1]).shape == (2, 1)


def test_too_small_input_rejected():
    with pytest.raises(ValueError, match="too small"):
        DCAE(latent_dim=8, input_size=64)


def test_unknown_archs():
    with pytest.raises(ValueError):
        build_classifier("vgg")
    with pytest.raises(ValueError):
        build_anomaly_model("rbm")


def test_shared_decoder_is_table_iv():
    for m in (DCAE(8), VAE(8), AAE(8)):
        d = m.decoder
        assert (d.deconv1.kernel_size, d.deconv2.stride, d.deconv3.kernel_size) == ((7, 7), (3, 3), (6, 6))
