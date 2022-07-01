import torch
from torch import nn
from transformers import AutoModel


class BertEmbedding(nn.Module):
    def __init__(self, bert_model_name: str, feature_mode: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.feature_mode = feature_mode

    def forward(
        self, input_ids, attention_mask
    ):
        """
        Args:
            tokens:
            fine_tune:
            feature_mode:
                1: extract the embedding of CLS
                2: average the embeddings of all pieces
        Returns:
        """
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).last_hidden_state
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, feat_dim, dtype=torch.float32).type_as(
            sequence_output
        )
        # only output the embedding of first sub-token
        for i in range(batch_size):
            # use pool to store the embeddings of pieces
            pool = []
            for j in range(max_len):
                if self.feature_mode == 1:
                    # for mode 1, only output the embedding of the CLS pieces
                    if j == 0:
                        valid_output[i] = sequence_output[i][j]
                        break
            #     elif feature_mode == 2:
            #         # for mode 2, average all the embedding of the pieces
            #         if sents_valid_loc[i][j].item() != 0:
            #             pool.append(sequence_output[i][j])
            # if feature_mode == 2:
            #     valid_output[i] = torch.mean(torch.stack(pool))
        return valid_output