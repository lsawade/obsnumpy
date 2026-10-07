"""A pipeline: a list of steps, declared once, applied to a whole array in one call."""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Pipeline:
    """Processing steps applied in order to the whole array.

    The array module follows the input: numpy in, numpy out; CuPy in, CuPy out.
    When every step is linear and acts on time, :meth:`matrix` compiles them
    into one ``(n_in, n_out)`` operator ``P``, so that applying them is one
    matrix product, ``y = x @ P``.
    """

    steps: list

    def __post_init__(self):
        self.steps = list(self.steps)

    def __call__(self, x, dt: float):
        """Run every step, in order. Returns ``(y, dt_out)``. An integer
        input is converted to float64 first."""
        if x.dtype.kind not in "fc":
            x = x.astype(np.float64)
        for s in self.steps:
            x = s(x, dt)
            dt = s.out_dt(dt)
        return x, dt

    def __len__(self):
        return len(self.steps)

    def split(self, index: int):
        """Two pipelines: the steps before ``index``, and the rest."""
        return Pipeline(self.steps[:index]), Pipeline(self.steps[index:])

    @property
    def linear(self) -> bool:
        """Whether every step is linear whatever the input. See :meth:`is_linear`
        for a given input."""
        return all(s.linear for s in self.steps)

    def _inputs(self, n_in, dt):
        """Each step's input: a list of ``(step, dt, npts)``."""
        out = []
        for s in self.steps:
            out.append((s, dt, n_in))
            n_in, dt = s.out_npts(n_in), s.out_dt(dt)
        return out

    def is_linear(self, n_in: int, dt: float) -> bool:
        """Whether every step is linear for an input of ``n_in`` samples at ``dt``."""
        return all(s.is_linear(d, n) for s, d, n in self._inputs(n_in, dt))

    def _require_linear(self, n_in, dt, what, time_only=False):
        bad = [type(s).__name__ for s, d, n in self._inputs(n_in, dt)
               if not s.is_linear(d, n) or (time_only and s.acts_on != "time")]
        if bad:
            kind = "linear steps that act on time" if time_only else "linear steps"
            raise ValueError(f"{what} needs {kind}; got {bad}")

    def adjoint(self, g, dt: float, n_in: int):
        """The adjoint of the whole pipeline, for an input of ``n_in`` samples
        at ``dt``: each step's adjoint, in reverse order. It maps a cotangent on
        the output back onto the input, ``<p(x), g> = <x, p.adjoint(g)>``."""
        self._require_linear(n_in, dt, "adjoint()")
        for s, d, n in reversed(self._inputs(n_in, dt)):
            g = s.adjoint(g, d, n)
        return g

    def out_dt(self, dt: float) -> float:
        """The sampling interval after every step."""
        for s in self.steps:
            dt = s.out_dt(dt)
        return dt

    def out_npts(self, npts: int) -> int:
        """The number of samples after every step."""
        for s in self.steps:
            npts = s.out_npts(npts)
        return npts

    def out_start(self, dt: float) -> float:
        """Seconds from the input's first sample to the output's first."""
        t = 0.0
        for s in self.steps:
            t += s.out_start(dt)
            dt = s.out_dt(dt)
        return t

    def out_components(self, components: list) -> list:
        """The component names after every step."""
        for s in self.steps:
            components = s.out_components(components)
        return components

    def matrix(self, n_in: int, dt: float, xp=np, dtype=np.float64):
        """The steps as one ``(n_in, n_out)`` matrix ``P``, so that ``y = x @ P``.

        It pushes the identity through the steps, so it costs one run of the
        pipeline on ``n_in`` traces, in ``n_in**2`` memory. It refuses steps
        that are nonlinear for this input or act on components.
        """
        self._require_linear(n_in, dt, "matrix()", time_only=True)
        P, _ = self(xp.eye(n_in, dtype=dtype), dt)
        return xp.ascontiguousarray(P)

    def compiled(self, n_in: int, dt: float, xp=np, dtype=np.float64):
        """A callable ``run(x)`` that applies the steps as ``x @ P`` (see
        :meth:`matrix`). It flattens the leading axes first, so the product is
        one 2-D GEMM. ``run.P`` is the matrix, ``run.dt`` the output's
        sampling interval, and ``run.adjoint(g)`` is ``g @ P.T``."""
        P = self.matrix(n_in, dt, xp, dtype)
        n_out = P.shape[1]

        def run(x):
            if x.shape[-1] != n_in:
                raise ValueError(f"compiled for {n_in} samples, got {x.shape[-1]}")
            return (x.reshape(-1, n_in) @ P).reshape(x.shape[:-1] + (n_out,))

        def adjoint(g):
            if g.shape[-1] != n_out:
                raise ValueError(f"compiled for {n_out} output samples, got {g.shape[-1]}")
            return (g.reshape(-1, n_out) @ P.T).reshape(g.shape[:-1] + (n_in,))

        run.P = P
        run.dt = self.out_dt(dt)
        run.adjoint = adjoint
        return run


def apply_to_dataset(dataset, pipeline: Pipeline):
    """Run ``pipeline`` on ``dataset.data`` and update its ``meta`` in place:
    ``delta``, ``npts``, ``starttime`` (from Cut and Interpolate) and
    ``components`` (from a rotation)."""
    meta = dataset.meta
    components = pipeline.out_components(list(meta.components))
    shift = pipeline.out_start(meta.delta)
    data, dt = pipeline(dataset.data, meta.delta)
    dataset.data = data
    meta.delta = dt
    meta.npts = data.shape[-1]
    meta.components = components
    if shift:
        meta.starttime = meta.starttime + shift
    return dataset
