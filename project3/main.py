import gymnasium as gym
import sys
import os
import shutil
import torch
from pathlib import Path

# ==========================================
# PART 1: AUTO-FIX ROMS (Run once check)
# ==========================================
def force_link_roms():
    """
    Ensures breakout.bin exists in the correct ALE folder.
    """
    try:
        import ale_py
        import AutoROM
        
        ale_path = Path(ale_py.__file__).parent / "roms"
        ale_path.mkdir(exist_ok=True)
        dest_file = ale_path / "breakout.bin"
        
        # Only run heavy checks if file is missing
        if not dest_file.exists():
            print("1. Game file missing. Attempting to find and link...")
            
            # Trigger Download
            from AutoROM import cli
            original_argv = sys.argv
            sys.argv = ["AutoROM", "--accept-license"]
            try:
                cli.main()
            except:
                pass
            finally:
                sys.argv = original_argv
            
            # Find and Copy
            autorom_path = Path(AutoROM.__file__).parent / "roms"
            found = list(autorom_path.rglob("breakout.bin"))
            
            if found:
                print(f"   Found game at: {found[0]}")
                shutil.copy(found[0], dest_file)
                print("   Game file restored successfully.")
            else:
                print("   CRITICAL: Could not find breakout.bin. Try running 'pip install autorom[accept-rom-license]'")
    except Exception as e:
        print(f"   Warning during setup: {e}")

# Run setup
force_link_roms()

# ==========================================
# PART 2: SETUP ENVIRONMENT
# ==========================================
import ale_py
gym.register_envs(ale_py) # Register Atari environments
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Import the class from your other file
from breakout import DQNBreakout

if __name__ == "__main__":
    print("2. Launching Game Window...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using Device: {device}")
    
    try:
        # Initialize the separated class
        env = DQNBreakout(render_mode="human", device=device)
        
        print("3. GAME ON! (Press Ctrl+C to stop)")
        
        state, info = env.reset()
        
        while True:
            # Random action (Replace with Agent later)
            action = env.action_space.sample()
            
            # Step
            state, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                state, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nGame Closed.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        try:
            env.close()
        except:
            pass