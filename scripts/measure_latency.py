"""Response time and coverage of a running service, for the report.

  python scripts/measure_latency.py http://localhost:8000            (or http://<ec2-address>)
Sends the demo moments to POST /recommend and prints median / 95th percentile time and how many got 3 areas.
"""
import json
import statistics
import sys
import time
import urllib.request

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")


def call(path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t) * 1000


moments, _ = call("/demo/moments?n=200")
times, full, none = [], 0, 0
for m in moments:
    r, ms = call("/recommend", {"hex": m["hex"], "time": m["time"], "minutes_empty": m["minutes_empty"]})
    times.append(ms)
    full += len(r["recommendations"]) == 3
    none += len(r["recommendations"]) == 0
times.sort()
print(f"{len(times)} requests: median {statistics.median(times):.0f} ms, 95th percentile {times[int(.95 * len(times)) - 1]:.0f} ms")
print(f"3 areas returned: {full / len(times):.1%}; 'not enough data': {none / len(times):.1%}")
