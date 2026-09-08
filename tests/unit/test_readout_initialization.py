import torch

from refsite_mlip.models.readout import SiteEnergyReadout


def test_zero_raw_branch_remains_trainable_and_mlp_propagates_gradient():
    with torch.random.fork_rng():
        torch.manual_seed(37)
        readout = SiteEnergyReadout(2, 3, 8, 1.0).double()
        scalars = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        central = torch.ones(4, 3, dtype=torch.float64)
        assert torch.count_nonzero(readout.raw.weight) == 0
        assert torch.count_nonzero(readout.raw(central)) == 0
        readout(scalars, central).sum().backward()
        assert torch.all(readout.raw.weight.grad == 4)
        assert torch.isfinite(scalars.grad).all()
        assert scalars.grad.norm() > 0
        assert readout.mlp[0].weight.grad.norm() > 0


def test_loading_existing_nonzero_raw_weights_preserves_predictions():
    with torch.random.fork_rng():
        torch.manual_seed(13)
        original = SiteEnergyReadout(2, 3, 8, 1.0).double()
        with torch.no_grad():
            original.raw.weight.fill_(0.25)
        scalars = torch.randn(4, 2, dtype=torch.float64)
        central = torch.randn(4, 3, dtype=torch.float64)
        restored = SiteEnergyReadout(2, 3, 8, 1.0).double()
        restored.load_state_dict(original.state_dict(), strict=True)
        assert torch.equal(original(scalars, central), restored(scalars, central))
