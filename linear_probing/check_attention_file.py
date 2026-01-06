import numpy as np

def check_attention_file_struc():
    
    # Load the file
    data = np.load('./eval_results/attention_per_layer/image_00000/q0_attention_per_layer.npz')

    # See all the keys (arrays) stored inside
    print("Keys in the file:")
    print(data.files)
    print(f"\nTotal keys: {len(data.files)}")

    # Check a specific layer
    print("\n=== Layer 0 data ===")
    print(f"mean_pooled_heads shape: {data['layer_0_mean_pooled_heads'].shape}")
    print(f"max_pooled_heads shape: {data['layer_0_max_pooled_heads'].shape}")
    print(f"mean_across_steps shape: {data['layer_0_mean_across_steps'].shape}")
    print(f"last_step shape: {data['layer_0_last_step'].shape}")

    # Check another layer
    print("\n=== Layer 35 data ===")
    print(f"mean_pooled_heads shape: {data['layer_35_mean_pooled_heads'].shape}")

    # Show all layer-specific keys
    layer_keys = [k for k in data.files if k.startswith('layer_')]
    print(f"\n=== All layer keys ({len(layer_keys)} total) ===")
    for key in sorted(layer_keys)[:10]:  # Show first 10
        print(f"  {key}: {data[key].shape}")
    print("  ...")

    data.close()
