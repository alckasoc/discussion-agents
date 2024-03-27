"""Fixtures for loading data-related assets."""

import os

import pytest


@pytest.fixture
def data_dir(pytestconfig) -> str:
    """Dir path to asset."""
    return os.path.join(pytestconfig.rootdir, "tests/assets")


@pytest.fixture
def hotpotqa_path(data_dir: str) -> str:
    """Dir path to HotPotQA data sample."""
    return os.path.join(data_dir, "hotpotqa")


@pytest.fixture
def hotpotqa_distractor_sample_path(hotpotqa_path: str) -> str:
    """Dir path to hotpotqa_distractor_sample path."""
    return os.path.join(hotpotqa_path, "hotpot-qa-distractor-sample.joblib")


@pytest.fixture
def expel_assets_path(data_dir: str) -> str:
    """Dir path to ExpeL assets."""
    return os.path.join(data_dir, "expel")


@pytest.fixture
def expel_experiences_10_fake_path(expel_assets_path: str) -> str:
    """Dir path to expel_experiences_10_fake experiences."""
    return os.path.join(expel_assets_path, "expel_experiences_10_fake.joblib")
