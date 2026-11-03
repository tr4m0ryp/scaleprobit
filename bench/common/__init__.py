"""Shared benchmark machinery: the data split record, the arms, the learning-curve runner, statistics."""
import warnings

from .split import Split, subsample_groups

warnings.filterwarnings("ignore", category=FutureWarning)
from .stats import arm_metrics, paired_summary

__all__ = ["Split", "arm_metrics", "paired_summary", "subsample_groups"]
