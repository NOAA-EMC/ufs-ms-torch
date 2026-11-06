# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Bounded delivery of trusted CPU setup shards, outside numerical transport."""

from __future__ import annotations

import hashlib
import tempfile

import torch
import torch.distributed as dist


def scatter_cpu_shards(output, payload=None, *, src=0, group=None):
    """Stream root's shards in 16 MiB messages with length and SHA-256 checks.

    Large object scatter stalls in the measured macOS Gloo setup path (G38).
    This retains the qualified benchmark's delivery protocol. It neither
    changes halo transport nor copies the global case to every GPU.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    if (src != 0 or len(output) != 1 or dist.get_backend(group) != "gloo"
            or dist.get_world_size(group) != world or dist.get_rank(group) != rank):
        raise ValueError("chunked setup scatter requires the complete Gloo world")
    chunk_bytes = 16 * 1024 * 1024
    failure = [None]
    if rank == src and (payload is None or len(payload) != world):
        failure[0] = "one cold-start shard is required per rank"
    dist.broadcast_object_list(failure, src=src, group=group)
    if failure[0] is not None:
        raise ValueError(failure[0])
    if rank == src:
        output[0] = payload[src]
    for destination in range(1, world):
        with tempfile.TemporaryFile() as archive:
            metadata = [None]
            if rank == src:
                try:
                    torch.save(payload[destination], archive)
                    size = archive.tell()
                    archive.seek(0)
                    digest = hashlib.file_digest(archive, "sha256").hexdigest()
                    archive.seek(0)
                    metadata[0] = {"bytes": size, "sha256": digest}
                except Exception as exc:
                    metadata[0] = {"error": f"{type(exc).__name__}: {exc}"}
            dist.broadcast_object_list(metadata, src=src, group=group)
            info = metadata[0]
            if "error" in info:
                raise ValueError(f"cold-start serialization failed: {info['error']}")
            size = info["bytes"]
            if not isinstance(size, int) or size <= 0:
                raise ValueError("invalid cold-start archive length")
            failure = None
            if rank in (src, destination):
                buffer = bytearray(min(chunk_bytes, size))
                view = memoryview(buffer)
                tensor = torch.frombuffer(buffer, dtype=torch.uint8)
                digest = hashlib.sha256()
                for offset in range(0, size, chunk_bytes):
                    count = min(chunk_bytes, size - offset)
                    if rank == src:
                        if archive.readinto(view[:count]) != count:
                            raise ValueError("truncated cold-start archive")
                        dist.send(tensor[:count], dst=destination, group=group)
                    else:
                        dist.recv(tensor[:count], src=src, group=group)
                        archive.write(view[:count])
                        digest.update(view[:count])
                if rank == destination:
                    try:
                        if digest.hexdigest() != info["sha256"]:
                            raise ValueError("cold-start archive SHA-256 mismatch")
                        archive.seek(0)
                        output[0] = torch.load(archive, map_location="cpu", weights_only=False)
                    except Exception as exc:
                        failure = f"rank {rank}: {type(exc).__name__}: {exc}"
            failures = [None] * world
            dist.all_gather_object(failures, failure, group=group)
            if any(item is not None for item in failures):
                raise ValueError(f"cold-start delivery failed: {failures}")
