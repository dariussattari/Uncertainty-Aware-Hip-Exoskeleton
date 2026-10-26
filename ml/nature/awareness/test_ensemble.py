"""The one check for the gait-phase ensemble baseline: heel strikes and phase from the insole force, the masked
ensemble loss, and the window-end target gather."""
import numpy as np
import torch

import ensemble_data as ed
import train_ensemble as te


def check() -> None:
    rate = 200
    # contact with hysteresis: a force hovering between the thresholds keeps its state; NaN carries it too
    force = np.array([0, 10, 60, 30, 30, np.nan, 10, 0, 60, 0], dtype = float)
    assert ed.get_contact(force).tolist() == [False, False, True, True, True, True, False, False, True, False]
    # heel strikes: three steps of 1.0 s, 1.2 s and 0.3 s (too short a stride) plus a bounce 0.05 s after a contact ends
    strikes = [100, 300, 540, 600]
    contact = np.zeros(1000, bool)
    for s in strikes:
        contact[s:s + 40] = True
    contact[350:360] = True                                   # a bounce: contact again 10 samples after the step at 300 ended
    found = ed.get_heel_strikes(contact, rate)
    assert found.tolist() == strikes, found
    phase = ed.get_phase(found, 1000, rate)
    assert np.isnan(phase[:100]).all() and np.isnan(phase[540:]).all()          # before the first stride, after the last kept one
    np.testing.assert_allclose(phase[[100, 200, 299]], [0, 0.5, 199 / 200], atol = 1e-6)   # 1.0 s stride: elapsed / duration
    np.testing.assert_allclose(phase[420], 0.5, atol = 1e-6)                    # the 1.2 s stride's midpoint
    assert np.isfinite(phase[100:540]).all() and np.isnan(phase[540:600]).all()   # the 0.3 s stride is dropped
    # masked loss: the mean of each member's masked MSE; an unlabelled leg (mask 0) counts for nothing
    preds = torch.tensor([[[1.0, 9.0], [0.0, 9.0]]])                           # (1 window, 2 members, 2 legs)
    y, mask = torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(te.get_masked_mse(preds, y, mask), torch.tensor(0.5))
    # the target is sin(2 pi phase) at the window's last sample, masked where the phase is undefined
    target = torch.tensor([[0.0, np.nan], [0.25, 0.5], [0.75, np.nan]])        # phase per sample and leg
    y, mask = te.get_targets(torch.sin(2 * np.pi * target), torch.tensor([0, 1]), window = 2)
    torch.testing.assert_close(y, torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    assert mask.tolist() == [[True, True], [True, False]]
    print('ok')


if __name__ == '__main__':
    check()
