import sys
import os
import subprocess

def run_step(step_name, script_name):
    print("\n" + "=" * 70)
    print(f" Executing Step: {step_name} ({script_name})")
    print("=" * 70)
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),script_name)
    if not os.path.exists(script_path):
       script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),script_name)
         
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
    print("\n🚀 Starting Part 2: Training a Tiny LLM Master Orchestrator")
    
    steps = [
        ("01 - Byte Tokenization & DataLoader", "1_dataset_dataloader.py"),
        ("02 - TinyLLM Architecture Assembly", "2_tiny_LLM.py"),
        ("03 - Loss Calculation & Evaluation", "3_loss_eval.py"),
        ("04 - AdamW Training Loop & Checkpointing", "4_training_loop.py"),
        ("05 - Autoregressive Text Generation", "5_text_generation.py"),
        ("06 - Part 2 Unit Tests Suite", "6_test.py")
    ]

    success_count = 0
    for title, script in steps:
        if run_step(title, script):
            success_count += 1
        else:
            print(f"\n⚠️ Pipeline halted due to failure in {script}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"🎉 PART 2 COMPLETE! Successfully executed {success_count}/{len(steps)} modules.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()