from argparse import ArgumentParser
import train.sagemaker_job_manager as sm_jobman 
from pprint import pprint
import os

def create_sagemaker_run_parser():

    parser = ArgumentParser(usage='Argument parser for sagemaker runs')
    # parser.add_argument('--network-type', type=str, help='Type of the network. hidden|unet-attn|unet-conv')
    parser.add_argument('--main-command', type=str, help='Type of the network')
    parser.add_argument('--train-folder', type=str, default=os.environ['SM_CHANNEL_TRAIN'], help='The input data directory')
    parser.add_argument('--val-folder', type=str, default=os.environ['SM_CHANNEL_TEST'], help='Validation images folder')
    parser.add_argument('--batch-size', '-b', type=int, help='The batch size.')
    parser.add_argument('--epochs', type=int, help='Number of epochs to run the simulation.')

    parser.add_argument('--name', default='', type=str, help='The name of the experiment.')
    # parser.add_argument('--job-name', default='$$timestamp--$$main-command--$$noise', type=str, help='String representation for the experiment name')
    parser.add_argument('--jobname-suffix', default='', type=str, help='Suffix to add to the job name')
    parser.add_argument('--job-folder', type=str, default=os.path.join(os.environ['SM_OUTPUT_DIR'], 'jobs'), help='The root folder of experiments')
    parser.add_argument('--model-folder', type=str, default=os.environ['SM_MODEL_DIR'], help='Trained model directory')
    parser.add_argument('--tensorboard-folder', type=str, default=os.path.join(os.environ['SM_OUTPUT_DIR'], 'tb-logs'), help='The root tensorboard folder')
    parser.add_argument('--checkpoint-folder', type=str, default='/opt/ml/checkpoints', help='The path where checkpoints are stored. Sagemaker syncs these to S3')

    parser.add_argument('--size', type=int, help='The size of the images (images are square so this is height and width).')
    parser.add_argument('--message', type=int, help='The length in bits of the watermark.')
    parser.add_argument('--tensorboard', type=int, default=1, help='Turn TensorBoard logging on/off.')
    # parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true', help='Enable mixed-precision training.')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='The device, cuda|cpu.')

    parser.add_argument('--enc-loss-weight', type=float, help='The weight of encoder loss in the overall loss function')
    parser.add_argument('--adv-loss-weight', type=float, help='The weight of the adversarial loss in the overall loss function')

    parser.add_argument('--encoder-blocks', type=int, help='Number of blocks in the encoder.')
    parser.add_argument('--encoder-channels', type=int, help='Number of inner channels in encoder blocks.')
    parser.add_argument('--encoder-block-type', choices=['Conv', 'Unet'], type=str, help='Encoder block type.')

    parser.add_argument('--decoder-blocks', type=int, help='Number of blocks in the decoder')
    parser.add_argument('--decoder-channels', type=int, help='Number of channels in decoder blocks.')
    parser.add_argument('--decoder-block-type', choices=['Conv', 'Unet'], type=str, help='Decoder block type.')

    parser.add_argument('--discriminator-blocks', type=int, help='Number of blocks in the discriminator.')
    parser.add_argument('--discriminator-channels', type=int, help='Number of channels in discriminator blocks.')
    parser.add_argument('--discriminator-block-type', choices=['Conv', 'Unet'], type=str, help='discriminator block type')

    parser.add_argument('--adam-lr', type=float, help='Learning rate of Adam')
    parser.add_argument('--noise', type=str, default='', required=False, help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout(0.55, 0.6)'")
    # parser.set_defaults(enable_fp16=False)

    return parser


def main():
    parser = create_sagemaker_run_parser()
    args = parser.parse_args()
    job = sm_jobman.SagemakerJobManager(args)
    print('-'*80)
    pprint(job.config)
    job.start_or_resume()


if __name__ == "__main__":
    main()