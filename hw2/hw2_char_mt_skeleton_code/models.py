import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
    ):

        super(Attention, self).__init__()
        "Luong et al. general attention (https://arxiv.org/pdf/1508.04025.pdf)"
        self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        query,
        encoder_enc_output,
        src_lengths,
    ):
        # query: (batch_size, 1, hidden_dim)
        # encoder_enc_output: (batch_size, max_src_len, hidden_dim)
        # src_lengths: (batch_size)
        # we will need to use this mask to assign float("-inf") in the attention scores
        # of the padding tokens (such that the output of the softmax is 0 in those positions)
        # Tip: use torch.masked_fill to do this
        # src_seq_mask: (batch_size, max_src_len)
        # the "~" is the elementwise NOT operator
        src_seq_mask = ~self.sequence_mask(src_lengths)
        #############################################
        # TODO: Implement the forward pass of the attention layer
        # Hints:
        # - Use torch.bmm to do the batch matrix multiplication
        #    (it does matrix multiplication for each sample in the batch)
        # - Use torch.softmax to do the softmax
        # - Use torch.tanh to do the tanh
        # - Use torch.masked_fill to do the masking of the padding tokens

        linear_encoding = self.linear_out(encoder_enc_output)

        linear_decoding = self.linear_in(query)


        activated = self.relu(linear_encoding + linear_decoding.unsqueeze(1))

        full_att_vec = self.full_att(activated)

        att_weights = self.softmax(full_att_vec)

        att_weights = att_weights.unsqueeze(1).squeeze(3)

        attn_out = torch.bmm(att_weights, encoder_enc_output)
        attn_out = attn_out.squeeze(1)


        # attn_out: (batch_size, 1, hidden_size)
        print(attn_out.shape)
        return attn_out

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (
            torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))
        )


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        '''
        src_vocab_size: train_dataset.input_lang.n_words
        hidden_size: opt.hidden_size
        padding_idx: PAD_IDX
        dropout: opt.dropout
        '''
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        '''
        src: (batch_size=64, max_src_len=19) source sentence
        lengths: (batch_size=64) (src != PAD_IDX).sum(1)
        '''
        # Convert word indexes to embeddings
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        
        # Pack padded batch of sequences for RNN module
        # pack padded sequences before passing them to the LSTM
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        # Forward pass through 
        # 1. enc_output: (batch_size=64, max_src_len=19, D=2 * hidden_size=64)
        # 2. final_hidden: tuple with 2 tensors (h_n and c_n)
        #    each tensor is (num_layers=1 * num_directions (D=2), batch_size=64, hidden_size=64)
        
        # output, (hn, cn) = rnn(input, (h0, c0))
        output, final_hidden = self.lstm(packed)
        
        # Unpack padding
        # unpack packed sequences after having passed them to the LSTM
        enc_output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Sum bidirectional reshape hidden
        # enc_output: (batch_size=64, max_src_len=19, hidden_size=64)
        enc_output = enc_output[:, :, :self.hidden_size] + enc_output[:, : ,self.hidden_size:]
        enc_output = self.dropout(enc_output)
        
        # Return output and final hidden state
        return enc_output, final_hidden

class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        '''
        hidden_size: opt.hidden_size
        tgt_vocab_size: train_dataset.output_lang.n_words
        attn: Attention(opt.hidden_size) or None
        padding_idx: PAD_IDX
        dropout: opt.dropout
        '''
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(
        self,
        tgt,
        dec_state,
        encoder_enc_output,
        src_lengths,
    ):
        '''
        tgt: 
            tgt_pred previous predicted element
            size: (batch_size=64, max_tgt_len=21) 
        
        dec_state: 
            final_enc_state (final_hidden of encoder)
            tuple with 2 tensors 
            each tensor has size: (num_layers=1 * num_directions (D=2), batch_size=64, hidden_size=64)
            
        encoder_enc_output: 
            encoder_outputs (enc_output of encoder)
            size: (batch_size=64, max_src_len=19, hidden_size=128); 
        
        src_lengths: 
            (src != PAD_IDX).sum(1)
            size: (batch_size) 
        '''
        if tgt.size(1) > 1:
            tgt = tgt[:, :-1]
        
        # bidirectional encoder enc_output are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)
        
        # dec_state[0]: final hidden state
        # dec_state[1]: final cell state
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        # # # # # # # # # #
        
        # we do the embedding of the current word
        embedded = self.embedding(tgt)
        embedded = self.dropout(embedded)
        
        # we do our calculations
        outputs, dec_state = self.lstm(embedded, dec_state)
        
        # the output has dropout applied
        outputs = self.dropout(outputs)
                
        # ... and we'll use the resulting decoder state in the next timestep
        
        # outputs: (batch_size=64, max_tgt_len=21, hidden_size=128)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers=1, batch_size=64, hidden_size=128)
        # print("outputs: ", outputs.shape)
        # print("dec_state[0]: ", dec_state[0].shape)
        return outputs, dec_state


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_enc_output, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_enc_output, src_lengths
        )

        return self.generator(output), dec_hidden
