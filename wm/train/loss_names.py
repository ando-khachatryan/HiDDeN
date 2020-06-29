import enum
# 'hidden_loss  ': g_loss.item(),
class LossNames(enum.Enum):
        network_loss  = 'network_loss' 
        encoder_mse   = 'encoder_mse'
        decoder_mse   = 'decoder_mse'
        bitwise       = 'bit-error'
        gen_adv_bce   = 'g_adv_bce'
        discr_cov_bce = 'd_cov_bce'
        discr_enc_bce = 'd_enc_bce'
        discr_avg_bce = 'd_avg_bce'

