"""RQ Worker 入口: python -m web.worker_entry"""

from __future__ import annotations

from redis import Redis
from rq import Connection, Worker

from web.core.saas_settings import get_redis_url


def main():
    redis_conn = Redis.from_url(get_redis_url())
    with Connection(redis_conn):
        Worker(["training"]).work()


if __name__ == "__main__":
    main()
