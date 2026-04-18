"""Optional weak-label and forest-mask logic for competition-oriented training.

Default training scripts use :func:`src.data_utils.load_tile_labels` unchanged.
Use :func:`load_tile_labels_enhanced` from the ``train_*_geo`` entrypoints only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.data_utils import TileLabels, reproject_to_match

logger = logging.getLogger(__name__)


def load_forest_mask_2020(
    forest_mask_dir: Path | None,
    tile_id: str,
    ref_profile: dict,
) -> np.ndarray | None:
    """Optional per-tile raster: True where forest in 2020."""
    if forest_mask_dir is None or not forest_mask_dir.is_dir():
        return None
    for name in (
        f"{tile_id}.tif",
        f"{tile_id}.tiff",
        f"forest_{tile_id}.tif",
        f"forest_{tile_id}.tiff",
    ):
        path = forest_mask_dir / name
        if path.is_file():
            arr = reproject_to_match(path, ref_profile)
            return (arr > 0).astype(bool)
    return None


def load_tile_labels_enhanced(
    data_dir: Path,
    tile_id: str,
    ref_profile: dict,
    *,
    year: int | None = None,
    forest_mask_dir: Path | None = None,
    label_strategy: str = "multi_source",
) -> TileLabels | None:
    """Weak labels with optional GLAD-L, forest mask, and multi-source voting.

    ``label_strategy``:
    - ``glads2_radd``: same rule as :func:`src.data_utils.load_tile_labels`.
    - ``multi_source``: 2-of-3 (GLAD-S2, RADD, GLAD-L for ``year``) for positives
      when GLAD-L rasters exist; negatives require no alert in all available sources.
    - ``xregion``: like ``multi_source`` but tolerates missing GLAD-S2 (Asia tiles).
      Uses majority of available sources (>=2 sources required, >=ceil(n/2) for positive,
      all-zero for negative). Tiles with fewer than 2 available sources return ``None``.
    """
    labels_dir = data_dir / "labels" / "train"

    glads2_alert_p = labels_dir / "glads2" / f"glads2_{tile_id}_alert.tif"
    glads2_date_p = labels_dir / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    radd_labels_p = labels_dir / "radd" / f"radd_{tile_id}_labels.tif"

    has_glads2 = glads2_alert_p.exists() and glads2_date_p.exists()
    has_radd = radd_labels_p.exists()

    if label_strategy in ("glads2_radd", "multi_source"):
        if not (has_glads2 and has_radd):
            return None
    elif label_strategy == "xregion":
        if not has_radd:
            # We always need RADD as the canonical date-aware label.
            return None
    else:
        raise ValueError(
            f"Unknown label_strategy={label_strategy!r}; use glads2_radd, multi_source, or xregion"
        )

    radd_r = reproject_to_match(radd_labels_p, ref_profile)
    radd_conf = radd_r // 10000
    radd_days = radd_r % 10000
    radd_date = np.datetime64("2014-12-31") + radd_days.astype("timedelta64[D]")
    radd_pos = (radd_conf >= 2) & (radd_date >= np.datetime64("2020-01-01"))

    if has_glads2:
        glads2_alert_r = reproject_to_match(glads2_alert_p, ref_profile)
        glads2_date_r = reproject_to_match(glads2_date_p, ref_profile)
        glads2_date_arr = np.datetime64("2019-01-01") + glads2_date_r.astype("timedelta64[D]")
        glads2_pos = (glads2_alert_r >= 2) & (glads2_date_arr >= np.datetime64("2020-01-01"))
    else:
        glads2_alert_r = None
        glads2_pos = None

    gladl_pos = None
    gladl_alert_r = None
    if year is not None:
        yy = year % 100
        gladl_alert_p = labels_dir / "gladl" / f"gladl_{tile_id}_alert{yy:02d}.tif"
        gladl_date_p = labels_dir / "gladl" / f"gladl_{tile_id}_alertDate{yy:02d}.tif"
        if gladl_alert_p.is_file() and gladl_date_p.is_file():
            gladl_alert_r = reproject_to_match(gladl_alert_p, ref_profile)
            gladl_date_r = reproject_to_match(gladl_date_p, ref_profile)
            gladl_date_arr = np.datetime64("2019-01-01") + gladl_date_r.astype("timedelta64[D]")
            gladl_pos = (gladl_alert_r >= 2) & (gladl_date_arr >= np.datetime64("2020-01-01"))

    if label_strategy == "glads2_radd":
        positive = glads2_pos & radd_pos
        negative = (glads2_alert_r == 0) & (radd_r == 0)
    elif label_strategy == "multi_source":
        if gladl_pos is not None:
            votes = (
                glads2_pos.astype(np.uint8)
                + radd_pos.astype(np.uint8)
                + gladl_pos.astype(np.uint8)
            )
            positive = votes >= 2
            negative = (glads2_alert_r == 0) & (radd_r == 0) & (gladl_alert_r == 0)
        else:
            positive = glads2_pos & radd_pos
            negative = (glads2_alert_r == 0) & (radd_r == 0)
    else:  # xregion
        pos_stack = [radd_pos]
        neg_stack = [(radd_r == 0)]
        if glads2_pos is not None:
            pos_stack.append(glads2_pos)
            neg_stack.append(glads2_alert_r == 0)
        if gladl_pos is not None:
            pos_stack.append(gladl_pos)
            neg_stack.append(gladl_alert_r == 0)
        if len(pos_stack) < 2:
            # Need at least two independent label sources to trust either polarity.
            return None
        votes = np.zeros_like(radd_pos, dtype=np.uint8)
        for pos in pos_stack:
            votes = votes + pos.astype(np.uint8)
        threshold = (len(pos_stack) + 1) // 2  # ceil(n/2)
        positive = votes >= threshold
        negative = neg_stack[0]
        for neg in neg_stack[1:]:
            negative = negative & neg

    forest = load_forest_mask_2020(forest_mask_dir, tile_id, ref_profile)
    if forest is not None:
        positive = positive & forest
        negative = negative & forest

    negative = negative & ~positive

    return TileLabels(positive=positive, negative=negative)


def label_tile_ids_xregion(labels_dir: Path) -> set[str]:
    """All tiles that have RADD plus at least one of (GLAD-S2 or GLAD-L any year)."""
    radd = {
        p.name.replace("radd_", "").replace("_labels.tif", "")
        for p in (labels_dir / "radd").glob("radd_*_labels.tif")
    }
    glads2_alert = {
        p.name.replace("glads2_", "").replace("_alert.tif", "")
        for p in (labels_dir / "glads2").glob("glads2_*_alert.tif")
    }
    glads2_date = {
        p.name.replace("glads2_", "").replace("_alertDate.tif", "")
        for p in (labels_dir / "glads2").glob("glads2_*_alertDate.tif")
    }
    glads2_full = glads2_alert & glads2_date
    gladl_any = set()
    gladl_dir = labels_dir / "gladl"
    if gladl_dir.is_dir():
        for p in gladl_dir.glob("gladl_*_alert*.tif"):
            stem = p.stem
            inner = stem[len("gladl_"):]
            # inner looks like '<tileid>_alert22' -> strip trailing '_alertNN'
            idx = inner.rfind("_alert")
            if idx > 0:
                gladl_any.add(inner[:idx])
    return radd & (glads2_full | gladl_any)
