from model.backbone.cnn import CNN
from model.seqmodel.seq2seq import Seq2Seq
from torch import nn

class VietOCR(nn.Module):
    def __init__(self, vocab_size,
                 backbone,
                 cnn_args, 
                 transformer_args, seq_modeling='transformer'):
        
        super(VietOCR, self).__init__()
        
        self.cnn = CNN(backbone, **cnn_args)
        self.seq_modeling = seq_modeling
        self.transformer = Seq2Seq(vocab_size, **transformer_args)


    def forward(self, img, tgt_input, tgt_key_padding_mask):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        src = self.cnn(img)
        outputs = self.transformer(src, tgt_input)

        return outputs