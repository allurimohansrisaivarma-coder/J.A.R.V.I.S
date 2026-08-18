import json
from pathlib import Path
import statistics

def analyze():
    log_file = Path("logs/performance.jsonl")
    if not log_file.exists():
        print("No logs found.")
        return
        
    ttfa = []
    ttft = []
    context = []
    total = []
    
    with open(log_file) as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if data.get("status") != "success": continue
                
                if data.get("ttfa_s") is not None:
                    ttfa.append(data["ttfa_s"])
                if data.get("ttft_s") is not None:
                    ttft.append(data["ttft_s"])
                if "stages" in data and "context" in data["stages"]:
                    context.append(data["stages"]["context"])
                if data.get("total_latency_s") is not None:
                    total.append(data["total_latency_s"])
            except:
                pass
                
    def print_stats(name, values):
        if not values:
            print(f"{name}: No data")
            return
        
        values.sort()
        p50 = statistics.median(values)
        p90 = values[int(len(values) * 0.9) if len(values) >= 10 else -1]
        p99 = values[int(len(values) * 0.99) if len(values) >= 100 else -1]
        print(f"{name} (N={len(values)}): P50={p50:.3f}s, P90={p90:.3f}s, P99={p99:.3f}s")
        
    print_stats("Context Gathering", context)
    print_stats("Time To First Token (TTFT)", ttft)
    print_stats("Time To First Audio (TTFA)", ttfa)
    print_stats("Total Latency", total)

if __name__ == "__main__":
    analyze()
