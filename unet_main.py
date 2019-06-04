from main_helper import prepare_training
from train import train


def main():
    model, device, network_config, train_options, this_run_folder, tb_logger = prepare_training('unet')
    train(model=model, device=device, train_options=train_options, this_run_folder=this_run_folder, tb_logger=tb_logger)


if __name__ == '__main__':
    main()
