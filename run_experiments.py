from main import main
import json
import time

results = {}
tenant_counts = [2,3,4]

print("Starting experiments...")

for n in tenant_counts:
    print(f"\n[Experiment] Running with {n} tenants...")
    start_time = time.time()
    
    # Run simulation
    # Using 1 experiment per setting for speed
    res = main(num_tenants=n, num_experiments=20)
    
    # Store results
    results[n] = res
    
    elapsed = time.time() - start_time
    print(f"[Experiment] {n} tenants finished in {elapsed:.2f}s")

# Save to JSON
with open('experiment_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nAll experiments finished. Results saved to experiment_results.json")
