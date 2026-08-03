"""DYNARTmo static articulatory model — midsagittal contour computation.

Ported (numpy-only, matplotlib stripped) from the supplementary notebook
`articulatoryModel.ipynb` of:

    Bernd J. Kröger, "DYNARTmo: A Dynamic Articulatory Model for Visualization
    of Speech Movement Patterns", arXiv:2507.20343.  Licensed CC-BY 4.0.
    https://arxiv.org/abs/2507.20343

The model stores a handful of extremal "anchor" vocal-tract contours (/i/, /a/,
/u/, /m/, /n/) plus a static bone/palate structure, and *interpolates* any
articulation from an 8-parameter articulator vector.  That is what lets every
French phoneme be drawn with no gaps: we only supply parameters, never per-phone
artwork.  See ``phone_params`` for the French IPA → parameter table.

Parameter vector (order matters): ``[toDors2, toDors1, toDors3, lips1, lips2,
toTip1, toTip2, velum]`` —

* ``toDors2``  tongue-body front(+1000) ↔ back(-1000)
* ``toDors1``  tongue-body high/close(+1000) ↔ low/open(-1000)
* ``toDors3``  tongue-dorsum closure (velar k/g/ŋ, 0..~1000)
* ``lips1``    lip closure (bilabial p/b/m, 0..1000)
* ``lips2``    lip rounding (spread 0 .. rounded 1000)
* ``toTip1``   tongue-tip closure (t/d/n/l/s/z, 0..1000)
* ``toTip2``   tongue-tip front ↔ back (-1000..1000)
* ``velum``    velum lowering for nasals (oral 0 .. nasal 1000)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).with_suffix("").parent / "data_sagi"


class SagittalStructure:
    """Static portions of the model (bones, palate) — constant across phones."""

    lens = [24, 24, 7, 6, 6, 6, 6, 6, 6, 6, 6, 20, 8, 6, 4, 8, 8, 8, 8]
    cutoffs = np.cumsum(lens)
    labels = ["hw1", "hw2", "hw3", "hw4", "hw5", "hw6", "hw7", "hw8", "hw9",
              "hw10", "hw11", "sc1", "sc2", "sc3", "sc4", "ok1", "ok2", "ok3",
              "na"]

    def __init__(self, data):
        self.data = data
        last = 0
        for label, cutoff in zip(self.labels, self.cutoffs):
            setattr(self, label, self.data[last:cutoff])
            last = cutoff


class SagittalContours:
    """Dynamic portions of the model (jaw, lips, velum, tongue)."""

    lens = [10, 10, 26, 4, 8, 16, 16, 34, 40]
    cutoffs = np.cumsum(lens)
    labels = ["jaw1", "jaw2", "jaw3", "larynx1", "larynx2",
              "lips1", "lips2", "velum", "tongue"]

    # anchor contours, populated once at import (class attributes ii/aa/uu/mm/nn)

    def __init__(self, data):
        if data.ndim == 2:
            data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        self.data = data
        last = 0
        for label, cutoff in zip(self.labels, self.cutoffs):
            setattr(self, label, self.data[last:cutoff])
            last = cutoff

    @classmethod
    def from_articulators(cls, art):
        jaw = np.arange(cls.cutoffs[2], dtype=int)
        lips = np.arange(cls.cutoffs[4], cls.cutoffs[6], dtype=int)
        jaw_and_lips = np.hstack([jaw, lips])

        # 1. Vocalic articulation: interpolate the i/u/a extremes.
        iiFac = np.tile(art.iiFac[..., np.newaxis], (sum(cls.lens), 1, 2))
        uuFac = np.tile(art.uuFac[..., np.newaxis], (sum(cls.lens), 1, 2))
        iiFac[jaw_and_lips] = np.hstack([art.iiFacLips.T, art.iiFacLips.T])
        uuFac[jaw_and_lips] = np.hstack([art.uuFacLips.T, art.uuFacLips.T])

        iuFac = np.hstack([art.iuFac.T, art.iuFac.T])
        aaFac = np.hstack([art.aaFac.T, art.aaFac.T])
        ret = cls(data=iuFac * (cls.ii.data * iiFac + cls.uu.data * uuFac)
                  + cls.aa.data * aaFac)

        low_teeth = np.array(ret.jaw1)

        # 2. Velum (nasal) articulation.
        velFac = np.hstack([art.velFac.T, art.velFac.T])
        ret.velum *= 1.0 - velFac
        ret.velum += cls.nn.velum * velFac

        # 3. Consonantal lip articulation (p/b/m).
        lipClosFac = np.hstack([art.lipClosFac.T, art.lipClosFac.T])
        ret.data[lips] *= 1.0 - lipClosFac
        ret.data[lips] += cls.mm.data[lips] * lipClosFac
        ret.data[jaw] *= 1.0 - lipClosFac * 0.66
        ret.data[jaw] += cls.mm.data[jaw] * lipClosFac * 0.66
        ret.lips2[0] = ret.jaw3[0]
        ret.lips2[-1] = ret.jaw1[-5]

        # 4. Co-elevation of tongue and lower jaw.
        low_teeth = ret.jaw1 - low_teeth
        ret.tongue[-5:] += low_teeth[-5:]
        ramp = np.linspace(0, 1, 10)[:, np.newaxis, np.newaxis]
        ret.tongue[-15:-5] += low_teeth * ramp

        # 5. Consonantal tongue-dorsum articulation (k/g).
        iuFacClipped = np.hstack([art.iuFacClipped.T, art.iuFacClipped.T])
        iiFacClipped = np.hstack([art.iiFacClipped.T, art.iiFacClipped.T])
        uuFacClipped = np.hstack([art.uuFacClipped.T, art.uuFacClipped.T])
        aaFacClipped = np.hstack([art.aaFacClipped.T, art.aaFacClipped.T])
        kk_tongue = (iuFacClipped * (cls.ii.tongue * iiFacClipped
                                     + cls.uu.tongue * uuFacClipped)
                     + cls.aa.tongue * aaFacClipped)
        fac = 0.5
        toDorsClosFac = np.hstack([art.toDorsClosFac.T, art.toDorsClosFac.T])
        ramp = np.hstack([np.linspace(0, 1, 13), np.ones(6),
                          np.linspace(1, fac, 9)])
        ramp = ramp[:, np.newaxis, np.newaxis]
        ret.tongue[12:40] *= 1.0 - toDorsClosFac * ramp
        ret.tongue[12:40] += kk_tongue[12:40] * toDorsClosFac * ramp
        for jl in ("jaw1", "jaw2", "jaw3", "lips2"):
            kk = (iuFacClipped * (getattr(cls.ii, jl) * iiFacClipped
                                  + getattr(cls.uu, jl) * uuFacClipped)
                  + getattr(cls.aa, jl) * aaFacClipped)
            arr = getattr(ret, jl)
            arr[:] *= 1.0 - toDorsClosFac * fac
            arr[:] += kk[:] * toDorsClosFac * fac

        # 6. Consonantal tongue-tip articulation (t/d/n/l).
        begin, end = 15, 37
        fac = 0.5
        start, mid = 32, 35
        high_toDors1_ramp = np.hstack([np.zeros(start - begin),
                                       np.linspace(0, 1, mid - start),
                                       np.ones(end - mid),
                                       np.linspace(1, 0, 40 - end)])
        start, mid = 15, 30
        low_toDors1_ramp = np.hstack([np.zeros(start - begin),
                                      np.linspace(0, 1, mid - start),
                                      np.ones(end - mid),
                                      np.linspace(1, fac, 40 - end)])
        ramp = np.tile(low_toDors1_ramp[:, np.newaxis],
                       (1, art.toDors1.shape[1]))
        ramp[:, art.toDors1[0] > 1000] = high_toDors1_ramp[:, np.newaxis]
        ramp = ramp[:, :, np.newaxis]
        toTipClosFac = np.hstack([art.toTipClosFac.T, art.toTipClosFac.T])
        ret.tongue[begin:] *= 1.0 - toTipClosFac * ramp
        ret.tongue[begin:] += cls.nn.tongue[start:] * toTipClosFac * ramp
        for jl in ("jaw1", "jaw2", "jaw3", "lips2"):
            arr = getattr(ret, jl)
            arr[:] *= 1.0 - toTipClosFac * fac
            arr[:] += getattr(cls.nn, jl)[:] * toTipClosFac * fac

        return ret


class Articulators:
    """8-parameter articulator vector (see module docstring for the order)."""

    lens = [1, 1, 1, 1, 1, 1, 1, 1]
    cutoffs = np.cumsum(lens)
    labels = ["toDors2", "toDors1", "toDors3", "lips1", "lips2",
              "toTip1", "toTip2", "velum"]

    def __init__(self, data):
        self.data = data[:, np.newaxis] if data.ndim == 1 else data
        last = 0
        for label, cutoff in zip(self.labels, self.cutoffs):
            setattr(self, label, self.data[last:cutoff])
            last = cutoff

    def __setattr__(self, key, value):
        if hasattr(self, key) and key in self.labels:
            getattr(self, key)[...] = value
        else:
            super().__setattr__(key, value)

    @property
    def iiFac(self):
        return (self.toDors2 + 1000.0) / 2000.0

    @property
    def uuFac(self):
        return 1.0 - self.iiFac

    @property
    def iiFacClipped(self):
        return (np.clip(self.toDors2, -500.0, 500.0) + 1000.0) / 2000.0

    @property
    def uuFacClipped(self):
        return 1.0 - self.iiFacClipped

    @property
    def iuFac(self):
        return (self.toDors1 + 1000.0) / 2000.0

    @property
    def aaFac(self):
        return 1.0 - self.iuFac

    @property
    def iuFacClipped(self):
        return np.ones_like(self.iiFac) * ((1400.0 + 1000.0) / 2000.0)

    @property
    def aaFacClipped(self):
        return 1.0 - self.iuFacClipped

    @property
    def uuFacLips(self):
        return self.lips2 / 1000.0

    @property
    def iiFacLips(self):
        return 1.0 - self.uuFacLips

    @property
    def lipClosFac(self):
        return self.lips1 / 1000.0

    @property
    def toTipClosFac(self):
        return self.toTip1 / 1000.0

    @property
    def toDorsClosFac(self):
        return self.toDors3 / 1000.0

    @property
    def velFac(self):
        return self.velum / 1000.0


_structure = SagittalStructure(np.loadtxt(_DATA_DIR / "structure.txt"))
for _name in ("ii", "aa", "uu", "mm", "nn"):
    setattr(SagittalContours, _name,
            SagittalContours(np.loadtxt(_DATA_DIR / f"{_name}.txt")))


# Order matters for a natural stroke order; "tongue" is flagged for highlighting.
_STRUCTURE_LABELS = list(SagittalStructure.labels)
_CONTOUR_LABELS = list(SagittalContours.labels)


def build_polylines(params: dict) -> tuple[list, list]:
    """Return ``(structure_lines, contour_lines)`` for an articulator dict.

    Each element is ``(points, is_tongue)`` where ``points`` is an ``(N, 2)``
    float array in model coordinates (y points up, matplotlib-style).
    """
    art = Articulators(np.zeros(8))
    for key, value in params.items():
        setattr(art, key, float(value))

    contour = SagittalContours.from_articulators(art)

    structure_lines = []
    for label in _STRUCTURE_LABELS:
        el = getattr(_structure, label)
        structure_lines.append((np.asarray(el, dtype=float), False))

    contour_lines = []
    for label in _CONTOUR_LABELS:
        el = getattr(contour, label)
        pts = np.column_stack([el[:, 0, 0], el[:, 0, 1]]).astype(float)
        contour_lines.append((pts, label == "tongue"))

    return structure_lines, contour_lines
