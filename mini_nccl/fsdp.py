"""Fully sharded data parallelism, built on this library's own collectives.

Where DDP replicates every parameter on every rank, FSDP splits them: rank
``r`` persistently holds only ``1/W`` of each sharded parameter. A unit's
full parameters exist only for the moment they are needed:

1. **Forward.** ``all_gather`` the unit's shards into a full flat buffer,
   point the module's parameters at views of it, run the forward, then
   release the buffer.
2. **Backward.** ``all_gather`` again, recompute the unit's forward with
   autograd enabled, take the gradients, ``reduce_scatter`` them so each
   rank receives exactly the gradient slice matching its parameter slice,
   then release again.

The optimizer only ever sees the local shards, so parameters, gradients, and
optimizer state all scale as ``1/W``. That is the trade FSDP makes: two
extra collectives per unit per step in exchange for a model that no longer
has to fit in one rank's memory.

**How this differs from PyTorch's FSDP.** Real FSDP re-gathers into the same
storage it freed, so tensors autograd saved during forward become valid again
when the storage is refilled. That is efficient but leans on storage-resize
internals. Here the unit's forward is instead *recomputed* in backward, the
same mechanism as activation checkpointing. It costs one extra forward per
unit and saves activation memory as a side effect, and it is far easier to
verify: nothing depends on a freed tensor still being reachable.

Parameters outside any unit (embeddings, the LM head, norms) stay replicated
and are all-reduced like DDP. This mirrors a real transformer wrap policy,
where only the repeated blocks are sharded, and it sidesteps sharding tied
weights.

Usage::

    fsdp = FullyShardedDataParallel(model, pg, unit_cls=Block)
    opt = torch.optim.AdamW(fsdp.parameters(), lr=3e-4)   # shards only

    for batch in loader:
        fsdp.zero_grad()
        loss = criterion(fsdp(batch))
        loss.backward()
        fsdp.sync()          # all-reduce the replicated gradients
        opt.step()

Each unit must take a single tensor and return a single tensor, which is the
shape of a transformer block.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from . import collectives
from .communicator import Communicator
from .ddp import average_gradients

_RELEASED = torch.empty(0)


class _Unit:
    """One module whose parameters are sharded across ranks."""

    def __init__(self, module: nn.Module, pg: Communicator, name: str) -> None:
        self.module = module
        self.pg = pg
        self.name = name
        self.owner: FullyShardedDataParallel | None = None

        # Record where each parameter lives so views can be reattached later.
        # Dedup by identity: a tied weight is one shard, reattached twice.
        self.entries: list[tuple[nn.Module, str, torch.Size, int]] = []
        self.params: list[nn.Parameter] = []
        seen: set[int] = set()
        for sub in module.modules():
            for attr, param in list(sub.named_parameters(recurse=False)):
                if id(param) in seen:
                    continue
                seen.add(id(param))
                self.entries.append((sub, attr, param.shape, param.numel()))
                self.params.append(param)

        dtypes = {p.dtype for p in self.params}
        if len(dtypes) > 1:
            raise ValueError(f"unit {name!r} mixes dtypes {dtypes}; shard them separately")
        self.dtype = self.params[0].dtype if self.params else torch.float32

        self.total_numel = sum(e[3] for e in self.entries)
        W = pg.world_size
        # reduce_scatter requires a length divisible by the world size, so the
        # flat buffer is padded; the tail lands on the last rank's shard.
        self.padded_numel = -(-self.total_numel // W) * W
        self.shard_numel = self.padded_numel // W

        flat = torch.zeros(self.padded_numel, dtype=self.dtype)
        offset = 0
        for param, (_, _, _, numel) in zip(self.params, self.entries, strict=True):
            flat[offset : offset + numel] = param.detach().reshape(-1)
            offset += numel
        start = pg.rank * self.shard_numel
        self.shard = nn.Parameter(flat[start : start + self.shard_numel].clone())

        self.release()

    # ---- materialization -------------------------------------------------

    @property
    def full_bytes(self) -> int:
        return self.padded_numel * self.shard.element_size()

    def materialize(self, shard: torch.Tensor) -> torch.Tensor:
        """all_gather the shards and point the module's parameters at them."""
        if self.pg.world_size == 1:
            flat = shard.detach().reshape(-1)
        else:
            flat = torch.cat(collectives.all_gather(self.pg, shard.detach()))
        offset = 0
        for (sub, attr, shape, numel) in self.entries:
            param = sub._parameters[attr]
            assert param is not None  # recorded from named_parameters, so present
            param.data = flat[offset : offset + numel].view(shape)
            offset += numel
        if self.owner is not None:
            self.owner._note_materialized(self.full_bytes)
        return flat

    def release(self) -> None:
        """Drop the full parameters, keeping only this rank's shard."""
        for (sub, attr, _, _) in self.entries:
            param = sub._parameters[attr]
            assert param is not None  # recorded from named_parameters, so present
            param.data = _RELEASED
        if self.owner is not None:
            self.owner._note_released(self.full_bytes)

    # ---- gradients -------------------------------------------------------

    def reduce_scatter_grads(self, grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
        """Flatten this unit's gradients and keep only this rank's slice."""
        flat = torch.zeros(self.padded_numel, dtype=self.dtype)
        offset = 0
        for grad, (_, _, _, numel) in zip(grads, self.entries, strict=True):
            if grad is not None:
                flat[offset : offset + numel] = grad.reshape(-1)
            offset += numel
        if self.pg.world_size == 1:
            return flat
        shard_grad = collectives.reduce_scatter(self.pg, flat, op="sum")
        return shard_grad.div_(self.pg.world_size)

    def gather_full_flat(self) -> torch.Tensor:
        if self.pg.world_size == 1:
            return self.shard.detach().reshape(-1)
        return torch.cat(collectives.all_gather(self.pg, self.shard.detach()))


class _ShardedForward(torch.autograd.Function):
    """Runs a unit's forward with its parameters gathered only transiently.

    Because backward recomputes the forward, the recompute has to reproduce it
    *exactly*, which means restoring the random number generator state. A unit
    containing dropout would otherwise draw a different mask the second time,
    and the gradients would be taken with respect to a graph that never
    produced the loss. The failure is silent and, worse, selectively invisible:
    layers downstream of the unit still get perfect gradients, so a smoke test
    that checks the head looks fine while the sharded weights are badly wrong.
    ``torch.utils.checkpoint`` handles this the same way and for the same
    reason.
    """

    @staticmethod
    def forward(ctx, unit: _Unit, shard: torch.Tensor, x: torch.Tensor):
        ctx.unit = unit
        ctx.save_for_backward(x, shard)
        ctx.cpu_rng_state = torch.get_rng_state()
        ctx.cuda_device = x.device if x.device.type == "cuda" else None
        if ctx.cuda_device is not None:
            ctx.cuda_rng_state = torch.cuda.get_rng_state(ctx.cuda_device)
        with torch.no_grad():
            unit.materialize(shard)
            try:
                y = unit.module(x)
            finally:
                unit.release()
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, shard = ctx.saved_tensors
        unit: _Unit = ctx.unit

        unit.materialize(shard)
        try:
            # Recompute this unit's forward, this time building a graph, so
            # gradients can be taken without having kept anything alive. The
            # generator state is rewound first so that any randomness inside
            # the unit replays identically, and fork_rng puts the outer stream
            # back afterwards so the caller's sequence is undisturbed.
            devices = [] if ctx.cuda_device is None else [ctx.cuda_device]
            x_local = x.detach().requires_grad_(True)
            with torch.random.fork_rng(devices=devices, enabled=True):
                torch.set_rng_state(ctx.cpu_rng_state)
                if ctx.cuda_device is not None:
                    torch.cuda.set_rng_state(ctx.cuda_rng_state, ctx.cuda_device)
                with torch.enable_grad():
                    y = unit.module(x_local)
            grads = torch.autograd.grad(
                y, [x_local, *unit.params], grad_outputs=grad_y, allow_unused=True
            )
            grad_x, param_grads = grads[0], grads[1:]
            shard_grad = unit.reduce_scatter_grads(param_grads)
        finally:
            unit.release()

        return None, shard_grad, grad_x if ctx.needs_input_grad[2] else None


class _UnitWrapper(nn.Module):
    """Stands in for a sharded module inside the model tree."""

    def __init__(self, unit: _Unit) -> None:
        super().__init__()
        self.unit = unit
        self.wrapped = unit.module
        self.shard = unit.shard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ShardedForward.apply(self.unit, self.shard, x)


class FullyShardedDataParallel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        pg: Communicator,
        unit_cls: type | tuple[type, ...] | None = None,
        units: list[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if unit_cls is None and units is None:
            raise ValueError("pass unit_cls or units to say what should be sharded")
        self.pg = pg
        self.units: list[_Unit] = []
        self._live_bytes = 0
        self.peak_materialized_bytes = 0

        # Names must be captured before wrapping, since inserting the unit
        # wrappers changes every sharded parameter's qualified path. Saved
        # checkpoints should look like the original model, not like FSDP's
        # internal tree.
        self._param_names = {id(p): name for name, p in module.named_parameters()}

        # Same weights everywhere before anything is sharded.
        if pg.world_size > 1:
            with torch.no_grad():
                for param in module.parameters():
                    collectives.broadcast(pg, param.data.contiguous(), src=0)
                for buffer in module.buffers():
                    collectives.broadcast(pg, buffer.data.contiguous(), src=0)

        targets = {id(m) for m in (units or [])}
        for parent in list(module.modules()):
            for key, child in list(parent._modules.items()):
                if child is None:
                    continue  # a registered-but-unset submodule slot
                selected = id(child) in targets or (
                    unit_cls is not None and isinstance(child, unit_cls)
                )
                if not selected or not any(True for _ in child.parameters()):
                    continue
                unit = _Unit(child, pg, name=f"{type(child).__name__.lower()}{len(self.units)}")
                unit.owner = self
                self.units.append(unit)
                parent._modules[key] = _UnitWrapper(unit)

        if not self.units:
            raise ValueError("no modules matched unit_cls/units, so nothing would be sharded")

        self.module = module
        self._unit_param_ids = {id(p) for unit in self.units for p in unit.params}
        shard_ids = {id(unit.shard) for unit in self.units}
        self._replicated = [
            p
            for p in module.parameters()
            if id(p) not in self._unit_param_ids and id(p) not in shard_ids
        ]

    # ---- what the optimizer should see -----------------------------------

    def shard_parameters(self) -> list[nn.Parameter]:
        return [unit.shard for unit in self.units]

    def parameters(self, recurse: bool = True):
        """Only the shards and the replicated parameters.

        The wrapped modules' parameters are placeholders pointing at gathered
        buffers that exist for microseconds at a time, so handing them to an
        optimizer would be a bug. Overriding this makes the obvious call
        (``optimizer(fsdp.parameters())``) the correct one.
        """
        return iter(self.shard_parameters() + self._replicated)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Matches the base signature, so anything in torch that calls it works.

        ``recurse`` and ``remove_duplicate`` are accepted and ignored: the list
        this returns is already flat and already deduplicated.
        """
        for unit in self.units:
            yield f"{prefix}{unit.name}.shard", unit.shard
        for param in self._replicated:
            # Original qualified names, so anything printing these (gradient
            # debugging, parameter groups) shows paths from the real model.
            yield prefix + self._param_names.get(id(param), "replicated"), param

    # ---- memory accounting ----------------------------------------------

    def _note_materialized(self, nbytes: int) -> None:
        self._live_bytes += nbytes
        self.peak_materialized_bytes = max(self.peak_materialized_bytes, self._live_bytes)

    def _note_released(self, nbytes: int) -> None:
        self._live_bytes = max(0, self._live_bytes - nbytes)

    def memory_report(self) -> dict[str, int]:
        """Bytes of parameter storage, measured rather than estimated."""
        element = self.units[0].shard.element_size()
        sharded_full = sum(u.padded_numel for u in self.units) * element
        shard_resident = sum(u.shard_numel for u in self.units) * element
        replicated = sum(p.numel() * p.element_size() for p in self._replicated)
        return {
            "sharded_params_full": sharded_full,
            "sharded_params_resident": shard_resident,
            "replicated_params": replicated,
            "peak_transient": self.peak_materialized_bytes,
            "ddp_equivalent": sharded_full + replicated,
            "fsdp_peak": shard_resident + replicated + self.peak_materialized_bytes,
        }

    # ---- training --------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync(self) -> None:
        """All-reduce the replicated gradients. Sharded ones are already done.

        Sharded gradients were reduce-scattered inside backward, so only the
        replicated parameters still need averaging.
        """
        if self.pg.world_size == 1 or not self._replicated:
            return
        average_gradients(self.pg, self._replicated)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.parameters():
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    # ---- checkpointing ---------------------------------------------------

    def full_state_dict(self) -> dict[str, torch.Tensor]:
        """Reassemble the unsharded model, which every rank can then save.

        Sharded parameters are gathered and unflattened back into their
        original shapes, so the result matches what the model would look like
        if FSDP had never touched it.
        """
        state: dict[str, torch.Tensor] = {}
        for unit in self.units:
            flat = unit.gather_full_flat()
            offset = 0
            for param, (_, _, shape, numel) in zip(unit.params, unit.entries, strict=True):
                name = self._param_names.get(id(param), f"{unit.name}.{offset}")
                state[name] = flat[offset : offset + numel].view(shape).clone()
                offset += numel
        for param in self._replicated:
            param_name = self._param_names.get(id(param))
            if param_name is not None:
                state[param_name] = param.detach().clone()
        for name, buffer in self.module.named_buffers():
            state[name] = buffer.detach().clone()
        return state
