"""Guardrails """

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src")
from build_features import haversine, norm_ein
from match_recipients import normalize


@pytest.mark.parametrize("raw,want", [
    (42103594, "042103594"),
    ("42103594", "042103594"),
    ("04-2103594", "042103594"),
    (" 042103594 ", "042103594"),
])
def test_ein_normalization(raw, want):
    """The join that silently fails if leading zeros are lost."""
    assert norm_ein(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("The Boston Foundation, Inc.", "BOSTON FOUNDATION"),
    ("BOSTON FOUNDATION", "BOSTON FOUNDATION"),
    ("Boston Foundation Incorporated", "BOSTON FOUNDATION"),
])
def test_name_normalization(raw, want):
    """Legal boilerplate must not defeat recipient matching."""
    assert normalize(raw) == want


def test_haversine_known_distance():
    """Boston to Worcester is about 60 km."""
    d = haversine(np.array([42.3601]), np.array([-71.0589]),
                  np.array([42.2626]), np.array([-71.8023]))
    assert 55 < d[0] < 70


def test_haversine_zero():
    d = haversine(np.array([42.0]), np.array([-71.0]),
                  np.array([42.0]), np.array([-71.0]))
    assert d[0] == pytest.approx(0, abs=1e-6)


def test_no_train_test_leakage():
    """No organization may appear in both splits."""
    tr = pd.read_csv("data/processed/pairs_train.csv", dtype=str)
    te = pd.read_csv("data/processed/pairs_test_full.csv", dtype=str)
    assert not set(tr["recipient_ein"]) & set(te["recipient_ein"])


def test_grants_are_sane():
    g = pd.read_csv("data/processed/grants.csv")
    assert len(g) > 1000
    assert (g["amount"] > 0).all()
    assert g["amount"].median() < 100_000        # long right tail, low median


def test_features_present():
    from core import FEATURES
    p = pd.read_csv("data/processed/pairs_train.csv")
    assert set(FEATURES).issubset(p.columns)


def test_no_individual_recipients_in_export():
    """Grants to named people must never reach the published dataset."""
    path = "app_data/grants_matched.csv"
    if not os.path.exists(path):
        pytest.skip("export not built")
    g = pd.read_csv(path)
    assert ("recipient_is_person" not in g.columns
            or g["recipient_is_person"].sum() == 0)