from transformer import *

if __name__ == '__main__':
    logger.setLevel(TensorLoggingLevels.enc_dec_block)
    emb = WordPositionEmbedding(1000)
    encoder = TransformerEncoder()
    decoder = TransformerDecoder()
    src_ids = torch.randint(1000, (5, 30))
    tgt_ids = torch.randint(1000, (5, 30))
    x = encoder(emb(src_ids))
    decoder(emb(tgt_ids), x)