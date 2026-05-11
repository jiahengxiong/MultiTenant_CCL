from __future__ import annotations

SUPPORTED_COLLECTIVES = frozenset(
    {"allgather", "allreduce", "reducescatter", "alltoall"}
)


def has_collective_workload(
    *,
    collective: str | None = None,
    single_flow_size: int | None = None,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
    tenant_collective_programs: dict[int, list[dict[str, object]]] | None = None,
) -> bool:
    """Return whether the workload can be generated from collective metadata."""

    return (
        tenant_collective_programs is not None
        or tenant_collective_specs is not None
        or (collective in SUPPORTED_COLLECTIVES and single_flow_size is not None)
    )


def normalize_collective_programs(
    tenant_mapping: dict[int, dict[int, int]],
    *,
    collective: str | None = None,
    single_flow_size: int | None = None,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
    tenant_collective_programs: dict[int, list[dict[str, object]]] | None = None,
) -> dict[int, list[dict[str, object]]] | None:
    """Normalize all collective workload forms into per-tenant programs.

    A single collective is represented as a one-op program. Existing explicit
    programs are copied so callers can mutate their local copy safely.
    """

    if tenant_collective_programs is not None:
        return {
            int(tenant): [dict(op) for op in program]
            for tenant, program in tenant_collective_programs.items()
        }

    if tenant_collective_specs is not None:
        programs = {}
        for tenant in tenant_mapping:
            if tenant not in tenant_collective_specs:
                raise ValueError(f"Missing collective spec for tenant {tenant}")
            spec = dict(tenant_collective_specs[tenant])
            programs[int(tenant)] = [
                {
                    "collective": spec.get("collective", collective),
                    "single_flow_size_bits": spec.get("single_flow_size_bits", single_flow_size),
                    "gap_after": float(spec.get("gap_after", 0.0)),
                }
            ]
        return programs

    if collective in SUPPORTED_COLLECTIVES and single_flow_size is not None:
        return {
            int(tenant): [
                {
                    "collective": collective,
                    "single_flow_size_bits": int(single_flow_size),
                    "gap_after": 0.0,
                }
            ]
            for tenant in tenant_mapping
        }

    return None
