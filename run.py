import argparse
import os
import random
import numpy as np
import torch
import torch.backends

from exp.exp_classification import Exp_Classification
from utils.print_args import print_args


# =========================
# Reproducibility
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return seed_worker, g


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prototype-based Time Series Classification')

    # ======================================================
    # Basic
    # ======================================================
    parser.add_argument('--task_name', type=str, default='classification')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='none')
    parser.add_argument('--model_id', type=str, default='')
    parser.add_argument('--model', type=str, default='')

    # ======================================================
    # Data
    # ======================================================
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--enc_in', type=int, default=2)
    parser.add_argument('--augmentation_ratio', type=float, default=0)
    # augmentation_ratio

    # ======================================================
    # Model (Backbone)
    # ======================================================
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)

    # ======================================================
    # Prototype Head (核心)
    # ======================================================
    parser.add_argument('--k_levels', type=int, nargs='+', default=[5, 3],
                        help='number of prototypes per class at each hierarchy')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--warm_up', type=int, default=3)

    # ======================================================
    # Frequency Module
    # ======================================================
    parser.add_argument('--use_fft_weight', action='store_true')
    parser.add_argument('--fft_weight_type', type=str, default='learnable',
                        choices=['learnable', 'lowpass', 'highpass'])

    # ======================================================
    # Optimization
    # ======================================================
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')

    # ======================================================
    # Runtime
    # ======================================================
    parser.add_argument('--seed', type=int, default=2025)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true')
    parser.add_argument('--devices', type=str, default='0')

    parser.add_argument('--des', type=str, default='exp')

    args = parser.parse_args()

    # ======================================================
    # Seed & Device
    # ======================================================
    seed_worker, generator = set_seed(args.seed)
    args.seed_worker = seed_worker
    args.generator = generator

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            args.device = torch.device("cpu")
        print('Using CPU / MPS')

    if args.use_gpu and args.use_multi_gpu:
        device_ids = args.devices.replace(' ', '').split(',')
        args.device_ids = [int(d) for d in device_ids]
        args.gpu = args.device_ids[0]

    print('================ Experiment Args ================')
    print_args(args)

    Exp = Exp_Classification

    # ======================================================
    # Train / Test
    # ======================================================
    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)

            setting = (
                f"{args.task_name}_"
                f"{args.model_id}_"
                f"{args.data}_"
                f"sl{args.seq_len}_"
                f"dm{args.d_model}_"
                f"nh{args.n_heads}_"
                f"el{args.e_layers}_"
                f"k{'-'.join(map(str, args.k_levels))}_"
                f"{args.des}_{ii}"
            )

            print(f">>>>>>> start training : {setting} >>>>>>>>>")
            exp.train(setting)

            print(f">>>>>>> testing : {setting} <<<<<<<<<")
            exp.test(setting)

            if args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
            elif args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()

    else:
        exp = Exp(args)
        setting = (
            f"{args.task_name}_"
            f"{args.model_id}_"
            f"{args.data}_"
            f"sl{args.seq_len}_"
            f"dm{args.d_model}_"
            f"k{'-'.join(map(str, args.k_levels))}_"
            f"{args.des}"
        )

        print(f">>>>>>> testing : {setting} <<<<<<<<<")
        exp.test(setting, test=1)
