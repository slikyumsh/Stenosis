from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import statistics
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse


def build_multipart_payload(
    image_path: Path,
    pixel_spacing_mm: float,
    confidence_threshold: float,
    render_artifacts: bool,
) -> tuple[bytes, str]:
    boundary = f"----stenosis-{uuid.uuid4().hex}"
    image_bytes = image_path.read_bytes()
    image_name = image_path.name
    content_type = mimetypes.guess_type(image_name)[0] or "application/octet-stream"

    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{image_name}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        image_bytes,
        b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="pixel_spacing_mm"\r\n\r\n',
        str(pixel_spacing_mm).encode(),
        b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="confidence_threshold"\r\n\r\n',
        str(confidence_threshold).encode(),
        b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="render_artifacts"\r\n\r\n',
        ("true" if render_artifacts else "false").encode(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def perform_request(target_url: str, body: bytes, content_type: str, timeout: int) -> dict:
    parsed = urlparse(target_url)
    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_cls(parsed.hostname, parsed.port, timeout=timeout)
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"

    started = time.perf_counter()
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        payload = response.read()
        elapsed = time.perf_counter() - started
        try:
            parsed_payload = json.loads(payload.decode("utf-8"))
        except Exception:
            parsed_payload = {"raw": payload.decode("utf-8", errors="replace")}
        return {
            "status_code": response.status,
            "elapsed_seconds": elapsed,
            "payload": parsed_payload,
        }
    finally:
        connection.close()


def _parse_mem_to_mib(mem_text: str) -> float:
    value = mem_text.strip().upper()
    suffixes = [
        ("GIB", 1024.0),
        ("MIB", 1.0),
        ("KIB", 1.0 / 1024),
        ("B", 1.0 / (1024 * 1024)),
    ]
    for suffix, multiplier in suffixes:
        if value.endswith(suffix):
            return float(value[: -len(suffix)].strip()) * multiplier
    return 0.0


def _sample_docker_stats(container_names: list[str]) -> dict[str, dict[str, float]]:
    command = (
        "docker stats --no-stream --format "
        "\"{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\" "
        + " ".join(container_names)
    )
    result = subprocess.run(
        ["wsl", "sh", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
    )
    stats: dict[str, dict[str, float]] = {}
    if result.returncode != 0:
        return stats

    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        name, cpu_text, mem_text = parts
        cpu_value = float(cpu_text.replace("%", "").replace(",", ".").strip() or 0.0)
        used_mem_text = mem_text.split("/")[0].strip()
        stats[name] = {
            "cpu_percent": cpu_value,
            "memory_mib": _parse_mem_to_mib(used_mem_text),
        }
    return stats


def _resource_sampler(
    stop_event: threading.Event,
    samples: list[dict],
    container_names: list[str],
    interval_seconds: float,
) -> None:
    while not stop_event.is_set():
        sample = _sample_docker_stats(container_names)
        if sample:
            samples.append(sample)
        stop_event.wait(interval_seconds)


def _summarize_resource_samples(samples: list[dict]) -> dict:
    per_container: dict[str, dict[str, list[float]]] = {}
    for sample in samples:
        for container_name, metrics in sample.items():
            bucket = per_container.setdefault(container_name, {"cpu": [], "memory": []})
            bucket["cpu"].append(float(metrics["cpu_percent"]))
            bucket["memory"].append(float(metrics["memory_mib"]))

    summary = {}
    for container_name, bucket in per_container.items():
        summary[container_name] = {
            "avg_cpu_percent": round(statistics.fmean(bucket["cpu"]), 3) if bucket["cpu"] else 0.0,
            "peak_cpu_percent": round(max(bucket["cpu"]), 3) if bucket["cpu"] else 0.0,
            "avg_memory_mib": round(statistics.fmean(bucket["memory"]), 3) if bucket["memory"] else 0.0,
            "peak_memory_mib": round(max(bucket["memory"]), 3) if bucket["memory"] else 0.0,
        }
    return summary


def run_scenario(
    target_url: str,
    image_path: Path,
    rps: float,
    requests_count: int,
    timeout: int,
    render_artifacts: bool,
    docker_targets: list[str],
    stats_interval_seconds: float,
) -> dict:
    body, content_type = build_multipart_payload(
        image_path,
        pixel_spacing_mm=1.0,
        confidence_threshold=0.25,
        render_artifacts=render_artifacts,
    )
    started = time.perf_counter()
    futures = []
    results = []
    lock = threading.Lock()

    stop_event = threading.Event()
    resource_samples: list[dict] = []
    sampler_thread = None
    if docker_targets:
        sampler_thread = threading.Thread(
            target=_resource_sampler,
            args=(stop_event, resource_samples, docker_targets, stats_interval_seconds),
            daemon=True,
        )
        sampler_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=max(2, min(requests_count, int(max(2, rps * 10))))) as executor:
            for index in range(requests_count):
                target_time = started + (index / rps)
                wait_seconds = target_time - time.perf_counter()
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                futures.append(executor.submit(perform_request, target_url, body, content_type, timeout))

            for future in as_completed(futures):
                result = future.result()
                with lock:
                    results.append(result)
    finally:
        stop_event.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=5)

    elapsed_values = [item["elapsed_seconds"] for item in results]
    ok_values = [item for item in results if item["status_code"] == 200]
    node_distribution: dict[str, int] = {}
    for item in ok_values:
        node_name = str(item["payload"].get("processing_node", "unknown"))
        node_distribution[node_name] = node_distribution.get(node_name, 0) + 1

    summary = {
        "rps": rps,
        "requests": requests_count,
        "render_artifacts": render_artifacts,
        "ok_requests": len(ok_values),
        "failed_requests": requests_count - len(ok_values),
        "mean_response_seconds": round(statistics.fmean(elapsed_values), 3) if elapsed_values else None,
        "min_response_seconds": round(min(elapsed_values), 3) if elapsed_values else None,
        "max_response_seconds": round(max(elapsed_values), 3) if elapsed_values else None,
        "status_codes": sorted({item["status_code"] for item in results}),
        "processing_nodes": node_distribution,
        "resource_usage": _summarize_resource_samples(resource_samples),
    }
    return {
        "summary": summary,
        "samples": results,
        "resource_samples_count": len(resource_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test the 2D analysis endpoint.")
    parser.add_argument("--url", default="http://localhost:8000/api/v1/analyze/2d")
    parser.add_argument("--image", required=True)
    parser.add_argument("--rps", nargs="+", type=float, default=[0.2, 0.5, 1.0])
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", default="load_test_2d_results.json")
    parser.add_argument("--render-artifacts", choices=["true", "false"], default="true")
    parser.add_argument(
        "--docker-targets",
        nargs="*",
        default=[],
        help="Optional docker container names to sample via docker stats during the load test.",
    )
    parser.add_argument("--stats-interval-seconds", type=float, default=1.0)
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    render_artifacts = args.render_artifacts.lower() == "true"
    report = {
        "target_url": args.url,
        "image": str(image_path),
        "requests_per_scenario": args.requests,
        "render_artifacts": render_artifacts,
        "docker_targets": args.docker_targets,
        "scenarios": [],
    }
    for rps in args.rps:
        scenario = run_scenario(
            args.url,
            image_path,
            rps,
            args.requests,
            args.timeout,
            render_artifacts,
            args.docker_targets,
            args.stats_interval_seconds,
        )
        report["scenarios"].append(scenario)
        summary = scenario["summary"]
        print(
            f"RPS={summary['rps']}: mean={summary['mean_response_seconds']}s, "
            f"min={summary['min_response_seconds']}s, max={summary['max_response_seconds']}s, "
            f"ok={summary['ok_requests']}/{summary['requests']}, nodes={summary['processing_nodes']}"
        )

    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
