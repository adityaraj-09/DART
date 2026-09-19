"""Paper eval and kill-tests."""

from dart.eval.andes import run_andes, run_push
from dart.eval.experiment import (
    andes_complete,
    cas_peer_hit,
    compare,
    grammar_ablation,
    idle_w0_forwards_flat,
    mesh_handover_suite,
    paper_suite,
    run_idd,
    run_workload,
    two_readers_cas,
    waiting_plugin_suite,
)

__all__ = [
    "andes_complete",
    "cas_peer_hit",
    "compare",
    "grammar_ablation",
    "idle_w0_forwards_flat",
    "mesh_handover_suite",
    "paper_suite",
    "run_andes",
    "run_idd",
    "run_push",
    "run_workload",
    "two_readers_cas",
    "waiting_plugin_suite",
]
