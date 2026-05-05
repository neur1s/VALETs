import os
import sys
import glob
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../include"))
from configuration_transfo_xl import TransfoXLConfig  # needed for unpickling

def save_hyperparams_from_runs(runs_dir='./runs'):
    wandb_dir = os.path.join(runs_dir, 'wandb')
    if not os.path.exists(wandb_dir):
        print(f"Directory not found: {wandb_dir}")
        return

    # look for run-* directories
    run_paths = glob.glob(os.path.join(wandb_dir, 'run-*'))
    print(f"Found {len(run_paths)} runs in {runs_dir}")

    for run_path in run_paths:
        files_dir = os.path.join(run_path, 'files')
        metrics_path = os.path.join(files_dir, 'val_wm_accuracy_valence.pt')
        checkpoint_path = os.path.join(files_dir, 'checkpoint.pt')
        output_path = os.path.join(files_dir, 'hyperparameters.txt')

        # only process if it looks like a valid run (has metrics)
        if not os.path.exists(metrics_path):
            print(f"Skipping {os.path.basename(run_path)} (incomplete run)")
            continue

        if os.path.exists(checkpoint_path):
            try:
                # load checkpoint
                # weights_only=False is required because we are leading complex objects (config)
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                if 'config' in checkpoint:
                    config = checkpoint['config']
                    
                    # handle both object and dict (just in case)
                    if hasattr(config, 'to_dict'):
                        config_dict = config.to_dict()
                    elif hasattr(config, '__dict__'):
                        config_dict = config.__dict__
                    elif isinstance(config, dict):
                        config_dict = config
                    else:
                        config_dict = {'unknown_config_type': str(config)}

                    # write to file
                    with open(output_path, 'w') as f:
                        for key, value in sorted(config_dict.items()):
                            f.write(f"{key}: {value}\n")
                    
                    print(f"Saved params for {os.path.basename(run_path)}")
                else:
                    print(f"No config found in checkpoint for {os.path.basename(run_path)}")

            except Exception as e:
                print(f"Error processing {os.path.basename(run_path)}: {e}")
        else:
            print(f"Checkpoint not found for {os.path.basename(run_path)}")

if __name__ == "__main__":
    save_hyperparams_from_runs()
