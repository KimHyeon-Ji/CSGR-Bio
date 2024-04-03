import argparse

def parser():
    parser = argparse.ArgumentParser(description='argparse')

    # Experiement setting
    parser.add_argument('--dataset', type=str, default='mit_bih_afib', choices=['mit_bih_afib', 'UniMiB_adl', 'MyoArmband'])
    parser.add_argument('--dataset-ver', type=int, default='1')
    parser.add_argument('--abl-ver', type=int, default='1')
    parser.add_argument('--data-split', type=str, default='random', choices=['subject_wise', 'random'])
    parser.add_argument('--task', type=str, default='single_model', choices=['single_model', 'rep_frozen', 'rep_fine_tuning'])
    parser.add_argument('--rep', type=str, default='supcon+intersubsim', choices=['supcon', 'intersubsim', 'supcon+intersubsim'])
    parser.add_argument('--sub-loss', type=str, default='Inter_sub_sim', choices=['Inter_sub_sim', 'Inter_sub_sim_v1', 'Inter_sub_sim_v2'])

    # Loss ratio
    parser.add_argument('--start-loss-ratio', type=float, default=0.3)
    parser.add_argument('--finish-loss-ratio', type=float, default=1)

    # Model setting
    parser.add_argument('--backbone-model', type=str, default='LSTM_FCN', choices=['LSTM_FCN'])

    # Dataset class & channel
    parser.add_argument('--mit-bih-afib-num-classes', type=int, default=4)
    parser.add_argument('--mit-bih-afib-num-features', type=int, default=2)
    parser.add_argument('--mit-bih-afib-window-size', type=int, default=2500)
    parser.add_argument('--mit-bih-afib-FCN-kernel-size', type=int, default=20)
    parser.add_argument('--mit-bih-afib-FCN-stride', type=int, default=3)

    parser.add_argument('--UniMiB-adl-num-classes', type=int, default=9)
    parser.add_argument('--UniMiB-adl-num-features', type=int, default=3)
    parser.add_argument('--UniMiB-adl-window-size', type=int, default=151)
    parser.add_argument('--UniMiB-adl-FCN-kernel-size', type=int, default=8)
    parser.add_argument('--UniMiB-adl-FCN-stride', type=int, default=1)

    parser.add_argument('--MyoArmband-num-classes', type=int, default=7)
    parser.add_argument('--MyoArmband-num-features', type=int, default=8)
    parser.add_argument('--MyoArmband-window-size', type=int, default=52)
    parser.add_argument('--MyoArmband-FCN-kernel-size', type=int, default=8)
    parser.add_argument('--MyoArmband-FCN-stride', type=int, default=1)

    # Data augmentation parameters
    parser.add_argument('--aug-scale', type=float, default=1)
    parser.add_argument('--aug-type', type=str, default='scaling', choices=['jitter', 'scaling'])
    parser.add_argument('--aug-max-seg', type=float, default=12)
    parser.add_argument('--aug-jitter-ratio', type=float, default=2)
    parser.add_argument('--aug-scaling-ratio', type=float, default=1.5)

    # LSTM FCN Model parameters
    parser.add_argument('--LSTM-FCN-output-dim', type=int, default=64)
    parser.add_argument('--LSTM-FCN-num-layer', type=int, default=1)
    parser.add_argument('--LSTM-FCN-lstm-drop-out', type=float, default=0.4)
    parser.add_argument('--LSTM-FCN-fc-drop-out', type=float, default=0.1)

    # Representation model parameter
    parser.add_argument('--Rep-head', type=str, default='mlp', choices=['linear', 'mlp'])
    parser.add_argument('--Rep-output-dim', type=int, default=32)

    # SupCon parameters
    parser.add_argument('--SupCon-temp', type=float, default=0.07)
    parser.add_argument('--SupCon-base-temp', type=float, default=0.07)
    parser.add_argument('--SupCon-contrast-mode', type=str, default='all', choices=['one', 'all'])

    # ProtoSim parameters
    parser.add_argument('--Intersubsim-temp', type=float, default=0.07)
    parser.add_argument('--Intersubsim-base-temp', type=float, default=0.07)
    parser.add_argument('--Intersubsim-dens-temp', type=float, default=0.2)

    # Optimizer parameters
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ds_lr', type=float, default=0.003)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    # Learning scheduler
    parser.add_argument('--lr-scheduler', type=str, default='LambdaLR',
                        choices=['LambdaLR', 'StepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts'])

    # Training hyperparameter
    parser.add_argument('--epochs', type=int, default=100, metavar='N')
    parser.add_argument('--batch-size', type=int, default=128)

    # Cuda, seed and logging
    parser.add_argument('--seed', type=int, default=2022, metavar='S')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52570)
    parser.add_argument("--host", default='127.0.0.1')

    return parser
