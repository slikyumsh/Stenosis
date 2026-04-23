from __future__ import annotations

from functools import lru_cache

from kafka import KafkaConsumer, KafkaProducer

from app.config import get_settings


@lru_cache(maxsize=1)
def get_kafka_producer() -> KafkaProducer:
    settings = get_settings()
    return KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        linger_ms=5,
        acks="all",
    )


@lru_cache(maxsize=1)
def get_kafka_consumer() -> KafkaConsumer:
    settings = get_settings()
    return KafkaConsumer(
        settings.kafka_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        enable_auto_commit=False,
        auto_offset_reset="latest",
        max_poll_records=1,
        value_deserializer=lambda value: value.decode("utf-8"),
    )


def enqueue_3d_job(job_id: str) -> None:
    settings = get_settings()
    producer = get_kafka_producer()
    producer.send(settings.kafka_topic, job_id.encode("utf-8"))
    producer.flush()


def blocking_pop_3d_job(timeout: int = 5) -> str | None:
    records = get_kafka_consumer().poll(timeout_ms=timeout * 1000, max_records=1)
    for partition_records in records.values():
        if partition_records:
            return str(partition_records[0].value)
    return None


def acknowledge_3d_job() -> None:
    get_kafka_consumer().commit()
