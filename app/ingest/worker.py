"""BullMQ worker: runs ingest in-process. The FastAPI process does not index."""

from __future__ import annotations

import asyncio
import os
import random
import signal
from datetime import datetime, timezone

import redis.asyncio as redis
from bullmq import Queue, Worker
from bullmq.custom_errors import UnrecoverableError
from dotenv import load_dotenv

load_dotenv()

from app.ingest.jobs import (  # noqa: E402
    emit_status,
    exception_message,
    is_indexable_document,
    load_file,
    run_ingest,
    update_status,
)

INDEXING_QUEUE = "indexing"
INDEXING_DLQ = "indexing-dlq"
INDEX_JOB_NAME = "index-document"
INDEX_MAX_ATTEMPTS = 3
INDEX_DLQ_MAX_ATTEMPTS = 5
INDEX_BACKOFF_TYPE = "exponential-jitter"
INDEX_DLQ_BACKOFF_TYPE = "exponential-jitter-dlq"
INDEX_LOCK_DURATION_MS = 10 * 60 * 1000

INDEX_CONCURRENCY = max(1, int(os.getenv("INDEX_CONCURRENCY", "3") or 3))
INDEX_CONCURRENCY_PER_USER = max(
    1, int(os.getenv("INDEX_CONCURRENCY_PER_USER", "2") or 2)
)


def _redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://127.0.0.1:6379").strip()


def _connection_opts() -> dict:
    return {
        "connection": _redis_url(),
        "concurrency": INDEX_CONCURRENCY,
        "lockDuration": INDEX_LOCK_DURATION_MS,
        "stalledInterval": 60_000,
        "settings": {"backoffStrategy": _backoff_strategy},
    }


def _backoff_strategy(attempts_made: int, type: str | None = None, *args) -> int:
    if type == INDEX_DLQ_BACKOFF_TYPE:
        base, cap = 30_000, 60 * 60 * 1000
    else:
        base, cap = 2_000, 15 * 60 * 1000
    exp = min(cap, base * (2 ** max(0, attempts_made - 1)))
    return round(exp * (0.5 + random.random()))


def _job_data(job) -> tuple[str, str]:
    data = job.data or {}
    file_id = data.get("fileId") or data.get("file_id")
    user_id = data.get("userId") or data.get("user_id")
    if not file_id or not user_id:
        raise UnrecoverableError("Job is missing fileId/userId")
    return str(file_id), str(user_id)


def _attempts_made(job) -> int:
    value = getattr(job, "attemptsMade", None)
    if value is None:
        value = getattr(job, "attempts_made", 0)
    return int(value or 0)


async def _acquire_user_slot(client: redis.Redis, user_id: str) -> None:
    key = f"droply:index:user:{user_id}"
    while True:
        n = int(await client.incr(key))
        if n == 1:
            await client.expire(key, 30 * 60)
        if n <= INDEX_CONCURRENCY_PER_USER:
            return
        await client.decr(key)
        await asyncio.sleep(1.5)


async def _release_user_slot(client: redis.Redis, user_id: str) -> None:
    key = f"droply:index:user:{user_id}"
    n = int(await client.decr(key))
    if n <= 0:
        await client.delete(key)


async def enqueue_dlq(file_id: str, user_id: str) -> None:
    queue = Queue(INDEXING_DLQ, {"connection": _redis_url()})
    try:
        await queue.add(
            INDEX_JOB_NAME,
            {"fileId": file_id, "userId": user_id},
            {
                "jobId": f"dlq-{file_id}",
                "attempts": INDEX_DLQ_MAX_ATTEMPTS,
                "backoff": {"type": INDEX_DLQ_BACKOFF_TYPE},
                "removeOnComplete": {"count": 1000},
                "removeOnFail": False,
            },
        )
    except Exception as exc:  # noqa: BLE001
        if "already" in str(exc).lower() and "exist" in str(exc).lower():
            return
        print(f"[indexing] failed to enqueue DLQ job: {exc}")
    finally:
        await queue.close()


async def process_index_job(job, max_attempts: int, slot_redis: redis.Redis) -> None:
    file_id, user_id = _job_data(job)
    attempt_number = _attempts_made(job) + 1

    await _acquire_user_slot(slot_redis, user_id)
    try:
        row = await asyncio.to_thread(load_file, file_id, user_id)
        if not row:
            raise UnrecoverableError(f"File not found: {file_id}")
        if row["is_trash"] or row["indexing_status"] == "COMPLETED":
            return
        if row["indexing_status"] == "INVALID" or not is_indexable_document(
            row["name"], row["type"], bool(row["is_folder"])
        ):
            return

        await asyncio.to_thread(
            update_status,
            file_id,
            user_id,
            indexing_status="INPROGRESS",
            index_error=None,
            increment_attempts=True,
        )
        await asyncio.to_thread(
            emit_status,
            file_id=file_id,
            user_id=user_id,
            indexing_status="INPROGRESS",
        )

        try:
            chunk_count = await asyncio.to_thread(run_ingest, row)
            indexed_at = datetime.now(timezone.utc)
            await asyncio.to_thread(
                update_status,
                file_id,
                user_id,
                indexing_status="COMPLETED",
                index_error=None,
                chunk_count=chunk_count,
                indexed_at=indexed_at,
            )
            await asyncio.to_thread(
                emit_status,
                file_id=file_id,
                user_id=user_id,
                indexing_status="COMPLETED",
                chunk_count=chunk_count,
                indexed_at=indexed_at,
            )
        except Exception as exc:
            if isinstance(exc, UnrecoverableError):
                raise
            message = exception_message(exc)
            terminal = attempt_number >= max_attempts
            status = "FAILED" if terminal else "PENDING"
            await asyncio.to_thread(
                update_status,
                file_id,
                user_id,
                indexing_status=status,
                index_error=message,
            )
            await asyncio.to_thread(
                emit_status,
                file_id=file_id,
                user_id=user_id,
                indexing_status=status,
                index_error=message,
            )
            raise RuntimeError(message) from exc
    finally:
        await _release_user_slot(slot_redis, user_id)


async def amain() -> None:
    slot_redis = redis.from_url(_redis_url(), decode_responses=True)
    opts = _connection_opts()
    dlq_opts = {
        **opts,
        "concurrency": min(INDEX_CONCURRENCY, 2),
    }

    async def process_main(job, _token):
        await process_index_job(job, INDEX_MAX_ATTEMPTS, slot_redis)

    async def process_dlq(job, _token):
        await process_index_job(job, INDEX_DLQ_MAX_ATTEMPTS, slot_redis)

    worker = Worker(INDEXING_QUEUE, process_main, opts)
    dlq_worker = Worker(INDEXING_DLQ, process_dlq, dlq_opts)

    def on_main_failed(job, err):
        if job is None:
            return
        if isinstance(err, UnrecoverableError):
            return
        attempts = _attempts_made(job)
        opts = getattr(job, "opts", None) or {}
        if isinstance(opts, dict):
            max_attempts = opts.get("attempts") or INDEX_MAX_ATTEMPTS
        else:
            max_attempts = getattr(opts, "attempts", None) or INDEX_MAX_ATTEMPTS
        if attempts < max_attempts:
            print(
                f"[indexing] retry file={job.data.get('fileId')} "
                f"attempt={attempts}: {err}"
            )
            return
        file_id, user_id = _job_data(job)
        print(f"[indexing] moving to DLQ file={file_id}: {err}")
        asyncio.create_task(enqueue_dlq(file_id, user_id))

    def on_dlq_failed(job, err):
        if job is None:
            return
        print(
            f"[indexing-dlq] failed file={(job.data or {}).get('fileId')} "
            f"attempt={_attempts_made(job)}: {err}"
        )

    worker.on("completed", lambda job, *_: print(
        f"[indexing] completed file={(job.data or {}).get('fileId')}"
    ))
    worker.on("failed", on_main_failed)
    worker.on("error", lambda err, *_: print(f"[indexing] worker error: {err}"))
    dlq_worker.on("completed", lambda job, *_: print(
        f"[indexing-dlq] completed file={(job.data or {}).get('fileId')}"
    ))
    dlq_worker.on("failed", on_dlq_failed)
    dlq_worker.on("error", lambda err, *_: print(f"[indexing-dlq] worker error: {err}"))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop(*_args):
        loop.call_soon_threadsafe(stop.set)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(
        f"Indexing workers started (queue + DLQ). "
        f"Redis {_redis_url()} concurrency={INDEX_CONCURRENCY}"
    )
    await stop.wait()
    print("Shutting down indexing workers…")
    await worker.close()
    await dlq_worker.close()
    await slot_redis.aclose()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
