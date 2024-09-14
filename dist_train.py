import os
import sys
import argparse

def setup_args():
    # Setup your command line arguments manually here
    sys.argv = [
        'recurrent_mix_precision_train.py',  # Simulate script name
        '-opt', '/root/IART/options/IART_REDS_N6_300K.yml',       # Configuration file path
        '--launcher', 'none',             # Launcher type
        '--auto_resume',                     # Include this flag if you want auto-resume
        # '--debug',                           # Include this flag to enable debug mode
        '--local_rank=0',                    # Local rank for distributed training
    ]

def main():
    setup_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # Set up PYTHONPATH as in the original bash script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = f"{parent_dir}:{current_pythonpath}"

    # Assuming train_pipeline and osp are from the same module or correctly imported
    from recurrent_mix_precision_train import train_pipeline
    import os.path as osp

    # Main entry logic
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)

if __name__ == "__main__":
    main()