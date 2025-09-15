import pandas as pd

# Load your results
df = pd.read_csv('ab_test_results.csv')

print("🔍 DETAILED ANALYSIS")
print("=" * 50)

# Check if differences are consistent across configurations
print("\n📊 Differences by configuration:")
for config in df[['array_size', 'precision']].drop_duplicates().values:
    size, precision = config
    subset = df[(df['array_size'] == size) & (df['precision'] == precision)]
    print(f"  {size}x{size} {precision}:")
    print(f"    Latency: {subset['latency_diff'].mean():.1f}ms")
    print(f"    Power: {subset['power_diff'].mean():.1f}W")
    print(f"    Memory: {subset['memory_diff'].mean():.1f}MB")

# Check relative differences
print("\n📈 Relative differences (%):")
df['latency_pct'] = (df['latency_diff'] / df['simple_latency']) * 100
df['power_pct'] = (df['power_diff'] / df['simple_power']) * 100
df['memory_pct'] = (df['memory_diff'] / df['simple_memory']) * 100

print(f"Latency: {df['latency_pct'].mean():.1f}% higher")
print(f"Power: {df['power_pct'].mean():.1f}% lower") 
print(f"Memory: {df['memory_pct'].mean():.1f}% higher")