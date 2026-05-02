from __future__ import annotations

import torch
import pytest

from shapesplat.frontend.dinov3_real import RealDINOv3Wrapper


class _DummyModel:
    patch_size = 16


def _wrapper() -> RealDINOv3Wrapper:
    wrapper = RealDINOv3Wrapper.__new__(RealDINOv3Wrapper)
    wrapper.cfg = {"frontend": {"dino_patch_size": 16, "dino_allow_global_feature_tiling": False}}
    wrapper.patch_size = 16
    wrapper.model = _DummyModel()
    wrapper.feature_layer = "last"
    wrapper.l2_normalize = False
    return wrapper


def test_tokens_to_feature_map_exact_grid():
    wrapper = _wrapper()
    tokens = torch.randn(1, 16, 384)
    fmap = wrapper._tokens_to_feature_map(tokens, image_hw=(64, 64), model_input_hw=(64, 64))
    assert list(fmap.shape) == [1, 384, 64, 64]


def test_tokens_to_feature_map_with_cls_register_tokens():
    wrapper = _wrapper()
    tokens = torch.randn(1, 1 + 4 + 16, 384)
    fmap = wrapper._tokens_to_feature_map(tokens, image_hw=(64, 64), model_input_hw=(64, 64))
    assert list(fmap.shape) == [1, 384, 64, 64]


def test_standardize_rejects_global_embedding():
    wrapper = _wrapper()
    with pytest.raises(ValueError, match="global image embedding.*dense patch features"):
        wrapper._standardize_dino_output(torch.randn(1, 384), image_hw=(64, 64), model_input_hw=(64, 64))


def test_standardize_dict_x_norm_patchtokens():
    wrapper = _wrapper()
    tokens = torch.randn(1, 16, 384)
    fmap = wrapper._standardize_dino_output({"x_norm_patchtokens": tokens}, image_hw=(64, 64), model_input_hw=(64, 64))
    assert list(fmap.shape) == [384, 64, 64]


def test_standardize_tuple_prefers_dense():
    wrapper = _wrapper()
    global_embedding = torch.randn(1, 384)
    dense_map = torch.randn(1, 384, 4, 4)
    fmap = wrapper._standardize_dino_output((global_embedding, dense_map), image_hw=(64, 64), model_input_hw=(64, 64))
    assert list(fmap.shape) == [384, 64, 64]
