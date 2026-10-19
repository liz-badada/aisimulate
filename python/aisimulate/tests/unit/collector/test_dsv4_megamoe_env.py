# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

_MODULE = "collector.sglang.collect_dsv4_megamoe"
# collect_dsv4_megamoe transitively imports dsv4_megamoe_workload at module level,
# so that module also gets cached with mock torch. Evict it too so subsequent tests
# that need real torch operations get a fresh import rather than the mock-polluted
# cached version.
_WORKLOAD_MODULE = "collector.sglang.dsv4_megamoe_workload"
_saved_module = sys.modules.pop(_MODULE, None)
_saved_workload_module = sys.modules.pop(_WORKLOAD_MODULE, None)
_saved_torch = sys.modules.get("torch")
_saved_torch_distributed = sys.modules.get("torch.distributed")
sys.modules["torch"] = MagicMock()
sys.modules["torch.distributed"] = MagicMock()
try:
    from collector.sglang.collect_dsv4_megamoe import (
        DEFAULT_MODEL_CONFIGS,
        CaseRunResult,
        MegaMoECase,
        aggregate_case_run_results,
        group_cases_for_logging,
        resolve_activation_clamp,
    )
finally:
    sys.modules.pop(_MODULE, None)
    sys.modules.pop(_WORKLOAD_MODULE, None)
    if _saved_module is not None:
        sys.modules[_MODULE] = _saved_module
    if _saved_workload_module is not None:
        sys.modules[_WORKLOAD_MODULE] = _saved_workload_module
    if _saved_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = _saved_torch
    if _saved_torch_distributed is None:
        sys.modules.pop("torch.distributed", None)
    else:
        sys.modules["torch.distributed"] = _saved_torch_distributed


def test_group_cases_for_logging_groups_seed_variants():
    cases = [
        MegaMoECase("context", 1024, "power_law_sampled_1.9", 8, 0),
        MegaMoECase("context", 1024, "power_law_sampled_1.9", 8, 1),
        MegaMoECase("context", 2048, "power_law_sampled_1.9", 8, 0),
    ]

    groups = group_cases_for_logging(cases)

    assert groups == [cases[:2], cases[2:]]


def test_aggregate_case_run_results_averages_latency_and_power():
    results = [
        CaseRunResult({"latency": "1.000000", "distribution": "power_law_sampled_1.9"}, {"power": 100.0}),
        CaseRunResult({"latency": "3.000000", "distribution": "power_law_sampled_1.9"}, {"power": 300.0}),
    ]

    aggregated = aggregate_case_run_results(results)

    assert aggregated.row["latency"] == "2.000000"
    assert aggregated.row["distribution"] == "power_law_sampled_1.9"
    assert aggregated.power_stats["power"] == 200.0


def test_qwen3_235b_config_matches_packaged_model_shape():
    from aisimulate_core.sdk.models.blocks import MoEBlockShape
    from aisimulate_core.sdk.models.helpers import _get_model_info

    cfg = DEFAULT_MODEL_CONFIGS["qwen3_235b"]
    shape = MoEBlockShape.from_model_info(_get_model_info(cfg["model"]))

    assert (cfg["hidden_size"], cfg["inter_size"]) == (shape.hidden_size, shape.moe_inter_size)
    assert (cfg["routed_num_experts"], cfg["routed_topk"]) == (shape.num_experts, shape.topk)
    assert shape.num_shared_experts == 0
    assert cfg["activation_clamp"] is None


@pytest.mark.parametrize(
    ("config_name", "requested", "expected"),
    [("dsv4_pro", None, 10.0), ("dsv4_pro", 5.0, 5.0), ("qwen3_235b", None, None), ("kimi_k3", 0.03125, 0.03125)],
)
def test_resolve_activation_clamp_keeps_declared_contract(config_name, requested, expected):
    assert resolve_activation_clamp(DEFAULT_MODEL_CONFIGS[config_name], requested, config_name=config_name) == expected


@pytest.mark.parametrize(("config_name", "requested"), [("qwen3_235b", 10.0), ("kimi_k3", 10.0)])
def test_resolve_activation_clamp_rejects_conflicting_override(config_name, requested):
    with pytest.raises(ValueError, match="activation contract"):
        resolve_activation_clamp(DEFAULT_MODEL_CONFIGS[config_name], requested, config_name=config_name)
