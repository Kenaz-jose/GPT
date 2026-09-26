import sys
import os
import argparse
import subprocess

def run_step(step_name, script_name):
    print("\n" + "=" * 70)
    print(f" Executing Step: {step_name} ({script_name})")
    print("=" * 70)
    
    script_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    script_name
    )  

    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        print(f"❌ ERROR in {script_name}:")
        print(result.stderr)
        return False
    print(f"✅ Completed: {step_name}")
    return True

def main():
    print("\n🚀 Starting Part 1: Core Transformer Architecture Master Orchestrator")
    
    steps = [
        ("1 - Pure NumPy Attention", "1_attention_numpy.py"),
        ("2 - Sinusoidal Positional Encoding", "2_positional_encoding.py"),
        ("3 - Multi-Head Attention Shapes & Heatmaps", "3_multihead_attention.py"),
        ("4 - Position-Wise Feed-Forward Network", "4_feedforward_network.py"),
        ("5 - Full Transformer Decoder Block", "5_block.py"),
        ("8 - Unit Tests Suite", "8_test.py")
    ]

    success_count = 0
    for title, script in steps:
        if run_step(title, script):
            success_count += 1
        else:
            print(f"\n⚠️ Pipeline halted due to failure in {script}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"🎉 PART 1 COMPLETE! Successfully executed {success_count}/{len(steps)} modules.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
