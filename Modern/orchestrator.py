import sys
import os
import subprocess

def run_step(step_name, script_name):
    print("\n" + "=" * 70)
    print(f" Executing Step: {step_name} ({script_name})")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, script_name)

    if not os.path.isfile(script_path):
        print(f"❌ ERROR: Script not found: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout.strip())

    if result.returncode != 0:
        print(f"❌ ERROR in {script_name}:")
        print(result.stderr)
        return False

    print(f"✅ Completed: {step_name}")
    return True

def main():
    print("\n🚀 Starting Part 3: Modernizing the Architecture Master Orchestrator")
    
    steps = [
        ("01 - RMSNorm", "rmsnorm.py"),
        ("02 - SwiGLU Feed-Forward Network", "swiglu.py"),
        ("03 - Rotary Positional Embeddings (RoPE)", "rope.py"),
        ("04 - Grouped-Query Attention & KV-Cache", "gqa_kvcache.py"),
        ("05 - Modern LLaMA Transformer Block", "block.py"),
        ("06 - Part 3 Unit Tests Suite", "test.py")
    ]

    success_count = 0
    for title, script in steps:
        if run_step(title, script):
            success_count += 1
        else:
            print(f"\n⚠️ Pipeline halted due to failure in {script}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"🎉 PART 3 COMPLETE! Successfully executed {success_count}/{len(steps)} modules.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()