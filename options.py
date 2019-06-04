class TrainingOptions:
    """
    Configuration options for the training
    """

    def __init__(self,
                 batch_size: int,
                 number_of_epochs: int,
                 train_folder: str, validation_folder: str, runs_folder: str,
                 start_epoch: int,
                 experiment_name: str,
                 image_height: int,
                 image_width: int):

        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.runs_folder = runs_folder
        self.start_epoch = start_epoch
        self.experiment_name = experiment_name
        self.image_height = image_height
        self.image_width = image_width


class HiDDenConfiguration:
    """
    The HiDDeN network configuration.
    """
    def __init__(self, message_length: int,
                 encoder_blocks: int, encoder_channels: int,
                 decoder_blocks: int, decoder_channels: int,
                 use_discriminator: bool,
                 discriminator_blocks: int, discriminator_channels: int,
                 decoder_loss: float,
                 encoder_loss: float,
                 adversarial_loss: float,
                 enable_fp16: bool = False):
        self.message_length = message_length
        self.encoder_blocks = encoder_blocks
        self.encoder_channels = encoder_channels
        self.use_discriminator = use_discriminator
        self.decoder_blocks = decoder_blocks
        self.decoder_channels = decoder_channels
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.decoder_loss = decoder_loss
        self.encoder_loss = encoder_loss
        self.adversarial_loss = adversarial_loss
        self.enable_fp16 = enable_fp16


class UnetConfiguaration():
    """
    Unet Network configuration
    """
    def __init__(self,
                 encoder_num_downs: int,
                 decoder_blocks: int,
                 discriminator_blocks: int,
                 message_length: int,
                 decoder_loss_weight: float,
                 encoder_loss_weight: float,
                 adversarial_loss_weight: float):
        self.num_downs = encoder_num_downs
        self.decoder_blocks = decoder_blocks
        self.discriminator_blocks = discriminator_blocks
        self.message_length = message_length
        self.decoder_loss_weight = decoder_loss_weight
        self.encoder_loss_weight = encoder_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
