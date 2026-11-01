"""Shared encoder with task-specific heads for multilingual PnC."""

from __future__ import annotations


def build_multitask_model(model_name: str, punctuation_labels: int, case_labels: int):
    """Create the model lazily so data utilities do not require PyTorch."""
    import torch.nn as nn
    from transformers import AutoModel

    class MultiTaskPnCModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name, token=False)
            hidden_size = self.encoder.config.hidden_size
            dropout_rate = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
            self.dropout = nn.Dropout(dropout_rate)
            self.punctuation_head = nn.Linear(hidden_size, punctuation_labels)
            self.case_head = nn.Linear(hidden_size, case_labels)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                inputs["token_type_ids"] = token_type_ids
            hidden_states = self.encoder(**inputs).last_hidden_state
            hidden_states = self.dropout(hidden_states)
            return self.punctuation_head(hidden_states), self.case_head(hidden_states)

    return MultiTaskPnCModel()
