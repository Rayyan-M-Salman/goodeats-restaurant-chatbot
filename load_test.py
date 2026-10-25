"""
Load tester for the Good Eats restaurant chatbot.

Purpose: send many concurrent/rapid requests to your OWN locally-running
instance of restaurant-chatbot.py, to see how it behaves under load
(response times, error rates, whether it falls over) so you can decide
what safeguards (rate limiting, concurrency caps, timeouts) to add.

Only point this at a server you own/run yourself (default: 127.0.0.1).

Usage examples:
    python load_test.py
    python load_test.py --url http://127.0.0.1:5000/chat --concurrency 20 --total 200
    python load_test.py --concurrency 50 --duration 30
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

SAMPLE_MESSAGES = [
    "What are your opening hours?",
    "Do you have any vegan dishes?",
    "What's on the menu today?",
    "Where are you located?",
    "Do you take reservations?",
    "What allergens are in your dishes?",
]


def build_payload(i):
    return {
        "messages": [
            {"role": "user", "content": SAMPLE_MESSAGES[i % len(SAMPLE_MESSAGES)]}
        ]
    }


def send_one(url, i, timeout):
    payload = build_payload(i)
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - start
        return {
            "ok": resp.status_code == 200,
            "status": resp.status_code,
            "elapsed": elapsed,
            "error": None,
        }
    except requests.RequestException as e:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "status": None,
            "elapsed": elapsed,
            "error": str(e),
        }


def run_fixed_count(url, concurrency, total, timeout):
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_one, url, i, timeout) for i in range(total)]
        for n, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            print(f"\r{n}/{total} requests completed", end="", flush=True)
    print()
    return results


def run_for_duration(url, concurrency, duration, timeout):
    results = []
    end_at = time.perf_counter() + duration
    i = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        in_flight = set()
        while time.perf_counter() < end_at or in_flight:
            while len(in_flight) < concurrency and time.perf_counter() < end_at:
                in_flight.add(pool.submit(send_one, url, i, timeout))
                i += 1
            done = {f for f in in_flight if f.done()}
            for f in done:
                results.append(f.result())
                in_flight.discard(f)
            print(f"\r{len(results)} requests completed", end="", flush=True)
            time.sleep(0.01)
    print()
    return results


def summarize(results, wall_time):
    total = len(results)
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    latencies = [r["elapsed"] for r in results]

    status_counts = {}
    for r in failed:
        key = r["status"] if r["status"] is not None else f"conn_error:{r['error']}"
        status_counts[key] = status_counts.get(key, 0) + 1

    print("\n--- Results ---")
    print(f"Total requests:     {total}")
    print(f"Succeeded (200):    {len(ok)}")
    print(f"Failed:             {len(failed)}")
    if total:
        print(f"Success rate:       {len(ok) / total * 100:.1f}%")
    print(f"Wall time:          {wall_time:.2f}s")
    if wall_time > 0:
        print(f"Throughput:         {total / wall_time:.2f} req/s")

    if latencies:
        latencies_sorted = sorted(latencies)
        p50 = statistics.median(latencies_sorted)
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95) - 1]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99) - 1]
        print(f"Latency avg:        {statistics.mean(latencies):.2f}s")
        print(f"Latency p50:        {p50:.2f}s")
        print(f"Latency p95:        {p95:.2f}s")
        print(f"Latency p99:        {p99:.2f}s")
        print(f"Latency max:        {max(latencies):.2f}s")

    if status_counts:
        print("\nFailure breakdown:")
        for key, count in sorted(status_counts.items(), key=lambda x: -x[1]):
            print(f"  {key}: {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000/chat",
                         help="Target endpoint (default: local chatbot)")
    parser.add_argument("--concurrency", type=int, default=10,
                         help="Number of concurrent in-flight requests")
    parser.add_argument("--total", type=int, default=100,
                         help="Total number of requests to send (fixed-count mode)")
    parser.add_argument("--duration", type=float, default=None,
                         help="If set, run for this many seconds instead of --total")
    parser.add_argument("--timeout", type=float, default=185.0,
                         help="Per-request timeout in seconds")
    parser.add_argument("--allow-remote", action="store_true",
                         help="Allow targeting a non-localhost URL (off by default)")
    args = parser.parse_args()

    host = urlparse(args.url).hostname
    if host not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
        print(
            f"Refusing to target '{host}': this tool is meant for load-testing "
            "your own local instance. Pass --allow-remote if you really own "
            "and have permission to load-test this host.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Target:        {args.url}")
    print(f"Concurrency:   {args.concurrency}")
    if args.duration:
        print(f"Duration:      {args.duration}s")
    else:
        print(f"Total reqs:    {args.total}")
    print("Starting...\n")

    start = time.perf_counter()
    if args.duration:
        results = run_for_duration(args.url, args.concurrency, args.duration, args.timeout)
    else:
        results = run_fixed_count(args.url, args.concurrency, args.total, args.timeout)
    wall_time = time.perf_counter() - start

    summarize(results, wall_time)


if __name__ == "__main__":
    main()
