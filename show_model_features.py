import joblib

# Load feature columns
features = joblib.load('artifact/plot_model_trainer/plot_feature_columns.pkl')

# Load model bundle
bundle = joblib.load('artifact/plot_model_trainer/plot_v3_production_model.pkl')

print("\n" + "=" * 80)
print("PLOT MODEL TRAINING FEATURES")
print("=" * 80)
print(f"\nTotal Features: {len(features)}")
print("\n" + "-" * 80)
print("FEATURE CATEGORIES:")
print("-" * 80)

# Categorize features
geographic = [f for f in features if f in ['latitude', 'longitude']]
road_distances = [f for f in features if 'distance' in f and 'km' in f]
property_flags = [f for f in features if f.startswith('is_') or f.startswith('has_')]
road_width = [f for f in features if 'road_width' in f]
usage_type = [f for f in features if f.startswith('usage_type_')]
facing = [f for f in features if f.startswith('facing_direction_')]
clusters = [f for f in features if f.startswith('c_')]
other = [f for f in features if f in ['circle_rate', 'log_plot_area', 'dist_to_center']]

print(f"\n1. Geographic Coordinates ({len(geographic)}):")
for f in geographic:
    print(f"   - {f}")

print(f"\n2. Road Distance Features ({len(road_distances)}):")
for f in road_distances:
    print(f"   - {f}")

print(f"\n3. Property Flags/Binary Features ({len(property_flags)}):")
for f in property_flags:
    print(f"   - {f}")

print(f"\n4. Road Width Categories ({len(road_width)}):")
for f in road_width:
    print(f"   - {f}")

print(f"\n5. Usage Type One-Hot Encoded ({len(usage_type)}):")
for f in usage_type:
    print(f"   - {f}")

print(f"\n6. Facing Direction One-Hot Encoded ({len(facing)}):")
for f in facing:
    print(f"   - {f}")

print(f"\n7. Spatial Cluster Features ({len(clusters)}):")
print(f"   - {len(clusters)} cluster one-hot features (c_0 to c_{len(clusters)-1})")

print(f"\n8. Other Numeric Features ({len(other)}):")
for f in other:
    print(f"   - {f}")

print("\n" + "-" * 80)
print("MODEL INFORMATION:")
print("-" * 80)
print(f"Model Type: {bundle.get('best_model_name', 'Unknown')}")
print(f"Target Variable: {bundle.get('target', 'Unknown')}")
print(f"Evaluation Metric: {bundle.get('evaluation_metric', 'Unknown')}")

print("\n" + "=" * 80)
print("COMPLETE FEATURE LIST:")
print("=" * 80)
for i, feat in enumerate(features, 1):
    print(f"{i:3d}. {feat}")

print("\n" + "=" * 80)
