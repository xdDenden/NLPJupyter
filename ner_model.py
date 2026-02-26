"""
ner_model.py
─────────────────────────────────────────────────────────────────────────────
A backbone-agnostic Transformer + CRF model for token classification.

Works with any HuggingFace encoder (RoBERTa, BERT, BioBERT, SciBERT, …).
To swap backbones, change BASE_MODEL in config.py — nothing else needs editing.
"""

import torch
import torch.nn as nn
from pathlib import Path
from transformers import PreTrainedModel, AutoConfig, AutoModel
from torchcrf import CRF  # pip install pytorch-crf


# ── BIO transition rules ───────────────────────────────────────────────────
# Tag indices:  O=0  B-CHEMICAL=1  B-DISEASE=2  I-DISEASE=3  I-CHEMICAL=4
# Pairs (src, dst) that should never occur in valid BIO sequences.
_ILLEGAL_TRANSITIONS = [
    (0, 3), (0, 4),  # O  → I-*          (must open with B-)
    (1, 3), (2, 4),  # B-X → I-Y         (cross-entity jump)
    (4, 3), (3, 4),  # I-X → I-Y         (cross-entity continuation)
]
_ILLEGAL_START_TAGS = [3, 4]  # Sequences cannot begin with I-*


class TransformerNERWithCRF(PreTrainedModel):
    """
    Drop-in replacement for RobertaWithCRF that works with *any* encoder.
    Now supports toggling the CRF layer via the `use_crf` config flag.
    """

    config_class = AutoConfig
    base_model_prefix = "backbone"

    # ── construction ──────────────────────────────────────────────────────
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Check if it's in config (for loaded models) or kwargs (for new models)
        self.use_crf = getattr(config, "use_crf", kwargs.get("use_crf", True))

        # Backbone — resolved from config, so any architecture works
        self.backbone = AutoModel.from_config(config)

        # Some models use 'hidden_dropout_prob' (BERT-family), others 'dropout'
        dropout_p = getattr(config, "hidden_dropout_prob",
                            getattr(config, "dropout", 0.1))
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Conditionally create the CRF or a standard loss function
        if self.use_crf:
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)
            print("--- Model initialized with CRF layer ---")
        else:
            self.loss_fct = nn.CrossEntropyLoss()
            print("--- Model initialized with Linear layer (No CRF) ---")

        self.post_init()

    # ── smart from_pretrained ─────────────────────────────────────────────
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        saved_path = Path(pretrained_model_name_or_path)
        is_checkpoint = (saved_path / "pytorch_model.bin").exists() or \
                        (saved_path / "model.safetensors").exists()

        if is_checkpoint:
            return super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

        # ── First-time init from a hub encoder ──
        num_labels = kwargs.pop("num_labels", 5)
        id2label = kwargs.pop("id2label", None)
        label2id = kwargs.pop("label2id", None)
        use_crf = kwargs.pop("use_crf", True)

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        # Manually inject use_crf into the config so it's saved to config.json
        config.use_crf = use_crf

        model = cls(config)

        # Copy encoder weights; the classifier head stays randomly initialised
        encoder = AutoModel.from_pretrained(pretrained_model_name_or_path, config=config)
        model.backbone.load_state_dict(encoder.state_dict())
        return model

    # ── BIO constraint enforcement ────────────────────────────────────────
    def _apply_transition_constraints(self):
        """Hard-penalise illegal BIO transitions so Viterbi avoids them."""
        if not self.use_crf:
            return

        with torch.no_grad():
            for src, dst in _ILLEGAL_TRANSITIONS:
                self.crf.transitions[src, dst] = -1e4
            for tag in _ILLEGAL_START_TAGS:
                self.crf.start_transitions[tag] = -1e4

    # ── shared encoder + emission step ───────────────────────────────────
    def _get_emissions(self, input_ids, attention_mask=None,
                       token_type_ids=None, **kwargs):
        """Run backbone → dropout → linear; return (B, T, num_labels)."""
        hidden = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )[0]  # take last hidden states only
        return self.classifier(self.dropout(hidden))

    # ── training forward ──────────────────────────────────────────────────
    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, labels=None, **kwargs):
        emissions = self._get_emissions(input_ids, attention_mask,
                                        token_type_ids, **kwargs)
        self._apply_transition_constraints()

        loss = None
        if labels is not None:
            if self.use_crf:
                mask = attention_mask.byte() if attention_mask is not None else None
                # HuggingFace uses -100 for padding; pytorch-crf crashes on it → zero-fill
                clean_labels = labels.clone()
                clean_labels[clean_labels == -100] = 0
                loss = -self.crf(emissions, clean_labels, mask=mask, reduction="mean")
            else:
                # Standard Token Classification CrossEntropyLoss
                loss = self.loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))

        return (loss, emissions) if loss is not None else (emissions,)

    # ── Viterbi / Argmax inference ────────────────────────────────────────
    def decode(self, input_ids, attention_mask=None):
        """Return best tag sequence per sample as List[List[int]]."""
        emissions = self._get_emissions(input_ids, attention_mask)
        mask = attention_mask.byte() if attention_mask is not None else None

        if self.use_crf:
            self._apply_transition_constraints()
            return self.crf.decode(emissions, mask=mask)
        else:
            # Simple Argmax for non-CRF mode
            preds = emissions.argmax(dim=-1)
            if mask is not None:
                return [[p for p, m in zip(pred_row, mask_row) if m]
                        for pred_row, mask_row in zip(preds.tolist(), mask.tolist())]
            return preds.tolist()