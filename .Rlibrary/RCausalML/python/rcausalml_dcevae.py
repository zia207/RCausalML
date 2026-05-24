"""
Bridge for RCausalML::DCEVAE(): Tabular DCEVAE (inst/dcevae/DCEVAE_ours).
Loads model + train from dcevae_ours_dir (package system.file path).
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import types

import numpy as np
import torch
import torch.utils.data as tud


class _SilentLogger:
    def info(self, *args, **kwargs):
        pass


def _make_loaders(r, d, a, y, batch_size, seed, valid_pct=0.2, test_pct=0.2):
    """Match Tabular DCEVAE_ours/utils.make_loader splits and paired shuffles."""
    np.random.seed(int(seed))
    r = np.asarray(r, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    if r.shape[0] != d.shape[0] or r.shape[0] != a.shape[0] or r.shape[0] != y.shape[0]:
        raise ValueError("r, d, a, y must have the same number of rows")

    n = a.shape[0]
    shuffle = np.random.permutation(n)
    valid_ct = int(n * valid_pct)
    test_ct = int(n * test_pct)
    valid_inds = shuffle[:valid_ct]
    test_inds = shuffle[valid_ct : valid_ct + test_ct]
    train_inds = shuffle[valid_ct + test_ct :]
    if train_inds.size == 0:
        raise ValueError(
            "Train split is empty: increase sample size or lower valid_pct/test_pct."
        )

    a_valid = a[valid_inds]
    r_valid = r[valid_inds]
    d_valid = d[valid_inds]
    y_valid = y[valid_inds]
    shuffle_v = np.random.permutation(a_valid.shape[0])
    a_valid2 = a_valid[shuffle_v]
    r_valid2 = r_valid[shuffle_v]
    d_valid2 = d_valid[shuffle_v]
    y_valid2 = y_valid[shuffle_v]

    a_test = a[test_inds]
    r_test = r[test_inds]
    d_test = d[test_inds]
    y_test = y[test_inds]
    shuffle_t = np.random.permutation(a_test.shape[0])
    a_test2 = a_test[shuffle_t]
    r_test2 = r_test[shuffle_t]
    d_test2 = d_test[shuffle_t]
    y_test2 = y_test[shuffle_t]

    a_train = a[train_inds]
    r_train = r[train_inds]
    d_train = d[train_inds]
    y_train = y[train_inds]
    shuffle_tr = np.random.permutation(a_train.shape[0])
    a_train2 = a_train[shuffle_tr]
    r_train2 = r_train[shuffle_tr]
    d_train2 = d_train[shuffle_tr]
    y_train2 = y_train[shuffle_tr]

    def _ds(rt, dt, at, yt, r2, d2, a2, y2):
        return tud.TensorDataset(
            torch.from_numpy(rt),
            torch.from_numpy(dt),
            torch.from_numpy(at),
            torch.from_numpy(yt),
            torch.from_numpy(r2),
            torch.from_numpy(d2),
            torch.from_numpy(a2),
            torch.from_numpy(y2),
        )

    train_set = _ds(r_train, d_train, a_train, y_train, r_train2, d_train2, a_train2, y_train2)
    valid_set = _ds(r_valid, d_valid, a_valid, y_valid, r_valid2, d_valid2, a_valid2, y_valid2)
    test_set = _ds(r_test, d_test, a_test, y_test, r_test2, d_test2, a_test2, y_test2)

    train_loader = tud.DataLoader(train_set, batch_size=int(batch_size), shuffle=True)
    valid_loader = tud.DataLoader(valid_set, batch_size=int(batch_size), shuffle=False)
    test_loader = tud.DataLoader(test_set, batch_size=int(batch_size), shuffle=False)

    input_dim = {
        "r": int(r_train.shape[1]),
        "d": int(d_train.shape[1]),
        "a": int(a_train.shape[1]),
        "y": int(y_train.shape[1]),
    }
    return train_loader, valid_loader, test_loader, input_dim


def fit_dcevae(
    r,
    d,
    a,
    y,
    dcevae_ours_dir,
    n_epochs=500,
    batch_size=256,
    lr=1e-4,
    loss_fn="BCE",
    break_epoch=30,
    act_fn="ReLU",
    a_y=1.0,
    a_r=1.0,
    a_d=1.0,
    a_a=1.0,
    a_f=0.0,
    a_h=0.4,
    u_kl=1.0,
    ur_dim=3,
    ud_dim=4,
    h_dim=100,
    seed=1,
    device=None,
    early_stop=True,
    save_plots=False,
):
    """
    Train DCEVAE (Tabular) and return (model, meta_dict).
    dcevae_ours_dir: directory containing model.py and train.py (DCEVAE_ours).
    """
    dcevae_ours_dir = os.path.abspath(dcevae_ours_dir)
    if dcevae_ours_dir not in sys.path:
        sys.path.insert(0, dcevae_ours_dir)
    from model import DCEVAE  # noqa: E402
    from train import train  # noqa: E402

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = tempfile.mkdtemp(prefix="rcausalml_dcevae_")
    args = types.SimpleNamespace(
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        loss_fn=str(loss_fn),
        break_epoch=int(break_epoch),
        act_fn=str(act_fn),
        a_y=float(a_y),
        a_r=float(a_r),
        a_d=float(a_d),
        a_a=float(a_a),
        a_f=float(a_f),
        a_h=float(a_h),
        u_kl=float(u_kl),
        ur_dim=int(ur_dim),
        ud_dim=int(ud_dim),
        h_dim=int(h_dim),
        seed=int(seed),
        device=str(device),
        early_stop=bool(early_stop),
        save_path=save_path,
        save_plots=bool(save_plots),
    )

    train_loader, valid_loader, _test_loader, input_dim = _make_loaders(
        r, d, a, y, args.batch_size, args.seed
    )

    model = DCEVAE(
        r_dim=input_dim["r"],
        d_dim=input_dim["d"],
        sens_dim=input_dim["a"],
        label_dim=input_dim["y"],
        args=args,
    ).to(args.device)

    logger = _SilentLogger()
    train(model, train_loader, valid_loader, args, logger)

    model_path = os.path.join(save_path, "model.pth")
    if not os.path.isfile(model_path):
        torch.save(model, model_path)

    try:
        loaded = torch.load(model_path, map_location=args.device, weights_only=False)
    except TypeError:
        loaded = torch.load(model_path, map_location=args.device)
    meta = {
        "save_path": save_path,
        "device": args.device,
        "input_dim": input_dim,
        "ur_dim": args.ur_dim,
        "ud_dim": args.ud_dim,
    }
    return loaded, meta


def predict_counterfactual_y_diff(model, r, d, a, y, device=None):
    """
    (sigmoid(y_cf) - sigmoid(y_f)) with sign flip by a, as in DCEVAE_ours/test.py.
    Stochastic due to reparameterization in q_u (same as evaluation in original code).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    r = np.asarray(r, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    with torch.no_grad():
        rt = torch.as_tensor(r, device=device, dtype=torch.float32)
        dt = torch.as_tensor(d, device=device, dtype=torch.float32)
        at = torch.as_tensor(a, device=device, dtype=torch.float32)
        yt = torch.as_tensor(y, device=device, dtype=torch.float32)
        u_mu, u_logvar = model.q_u(rt, dt, at, yt)
        u = model.reparameterize(u_mu, u_logvar)
        ur, ud = torch.split(u, [model.ur_dim, model.ud_dim], 1)
        _r_mu, _d_mu, y_p, _d_cf, y_p_cf = model.p_i(ur, ud, at)
        y_p_sig = torch.sigmoid(y_p)
        y_cf_sig = torch.sigmoid(y_p_cf)
        mask_a = torch.where(at == 1, -1.0, 1.0)
        cf = (y_cf_sig - y_p_sig) * mask_a
        return cf.cpu().numpy().ravel()
