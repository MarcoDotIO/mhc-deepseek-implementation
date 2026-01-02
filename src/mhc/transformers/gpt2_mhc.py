from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import inspect

from transformers import GPT2Config, GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import create_masks_for_generate
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP

from mhc.mhc import MhcProjector
from mhc.stream_ops import mhc_update, stream_weighted_sum


class MhcGPT2Config(GPT2Config):
    model_type = "mhc-gpt2"

    def __init__(
        self,
        *,
        mhc_n: int = 4,
        mhc_tmax: int = 20,
        mhc_alpha_init: float = 0.01,
        mhc_rmsnorm_eps: float = 1e-6,
        mhc_stream_init: str = "paper",
        mhc_readout_init: str = "first",
        **kwargs,
    ):
        """
        GPT-2 config with additional mHC fields.

        Args:
            mhc_n: expansion rate n (streams), paper commonly uses n=4.
            mhc_tmax: Sinkhorn-Knopp iterations (paper uses 20).
            mhc_alpha_init: init for gating scalars α_* (paper table uses 0.01).
            mhc_rmsnorm_eps: epsilon for coefficient RMSNorm.
            mhc_stream_init:
              - \"paper\": stream = (x, 0, ..., 0) (see paper Section 3 prelim)
              - \"copy\":  stream = (x, x, ..., x) (useful for exact GPT-2 equivalence init)
            mhc_readout_init:
              - \"first\": read out stream 0
              - \"mean\":  uniform average over streams
        """
        super().__init__(**kwargs)
        self.mhc_n = int(mhc_n)
        self.mhc_tmax = int(mhc_tmax)
        self.mhc_alpha_init = float(mhc_alpha_init)
        self.mhc_rmsnorm_eps = float(mhc_rmsnorm_eps)
        self.mhc_stream_init = str(mhc_stream_init)
        self.mhc_readout_init = str(mhc_readout_init)


class MhcGPT2Block(nn.Module):
    def __init__(self, config: MhcGPT2Config, *, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)

        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=self.layer_idx)

        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.mlp = GPT2MLP(inner_dim, config)

        # Two mHC steps per block: attention residual, then MLP residual.
        self.mhc_attn = MhcProjector(
            n_streams=config.mhc_n,
            hidden_dim=config.n_embd,
            tmax=config.mhc_tmax,
            alpha_init=config.mhc_alpha_init,
            rmsnorm_eps=config.mhc_rmsnorm_eps,
        )
        self.mhc_mlp = MhcProjector(
            n_streams=config.mhc_n,
            hidden_dim=config.n_embd,
            tmax=config.mhc_tmax,
            alpha_init=config.mhc_alpha_init,
            rmsnorm_eps=config.mhc_rmsnorm_eps,
        )

        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        x_stream: torch.Tensor,  # (B,T,n,C)
        *,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Cache], Optional[torch.Tensor]]:
        # Attention step
        maps = self.mhc_attn(x_stream)
        x_in = stream_weighted_sum(x_stream, maps.h_pre)
        x_in = self.ln_1(x_in)

        # transformers>=4.5x: GPT2Attention returns (attn_output, attn_weights) and updates
        # the `Cache` object in-place when provided.
        attn_output, attn_weights = self.attn(
            x_in,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        attn_output = self.resid_dropout(attn_output)
        x_stream = mhc_update(x_stream, h_post=maps.h_post, h_res=maps.h_res, y=attn_output)

        # MLP step
        maps2 = self.mhc_mlp(x_stream)
        x_in2 = stream_weighted_sum(x_stream, maps2.h_pre)
        x_in2 = self.ln_2(x_in2)
        mlp_output = self.mlp(x_in2)
        mlp_output = self.resid_dropout(mlp_output)
        x_stream = mhc_update(x_stream, h_post=maps2.h_post, h_res=maps2.h_res, y=mlp_output)

        # Cache is updated in-place; return the same cache object for the model loop.
        # If caching is disabled, keep returning None.
        return x_stream, past_key_values if use_cache else None, attn_weights if output_attentions else None


class MhcGPT2PreTrainedModel(PreTrainedModel):
    config_class = MhcGPT2Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        # We checkpoint at the block level in MhcGPT2Model.forward
        module.gradient_checkpointing = value


class MhcGPT2Model(MhcGPT2PreTrainedModel):
    def __init__(self, config: MhcGPT2Config):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([MhcGPT2Block(config, layer_idx=i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Learnable readout weights over streams (simplex via softmax).
        self.mhc_readout_logits = nn.Parameter(torch.zeros(config.mhc_n))

        self.post_init()
        self._init_readout()

    def _init_readout(self) -> None:
        with torch.no_grad():
            if self.config.mhc_readout_init == "mean":
                self.mhc_readout_logits.zero_()
            else:
                # "first": make softmax put mass on index 0.
                self.mhc_readout_logits.fill_(-5.0)
                self.mhc_readout_logits[0] = 5.0

    def _stream_init(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B,T,C) -> x_stream: (B,T,n,C)
        b, t, c = hidden_states.shape
        n = self.config.mhc_n
        if self.config.mhc_stream_init == "copy":
            return hidden_states.unsqueeze(-2).expand(b, t, n, c).contiguous()
        # "paper": (x, 0, ..., 0)
        x_stream = hidden_states.new_zeros((b, t, n, c))
        x_stream[:, :, 0, :] = hidden_states
        return x_stream

    def _readout(self, x_stream: torch.Tensor) -> torch.Tensor:
        # x_stream: (B,T,n,C) -> (B,T,C)
        w = torch.softmax(self.mhc_readout_logits, dim=0).to(dtype=x_stream.dtype)
        return torch.einsum("n,btnc->btc", w, x_stream)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if token_type_ids is not None:
            raise NotImplementedError("token_type_ids not supported for MhcGPT2Model")
        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise NotImplementedError("cross-attention not supported for MhcGPT2Model")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify only one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            input_shape = input_ids.size()
            inputs_embeds = self.wte(input_ids)
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape
        device = inputs_embeds.device

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + seq_length, device=device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # Attention mask: use PreTrainedModel helpers.
        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device=device)

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        hidden_states = inputs_embeds + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        x_stream = self._stream_init(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (self._readout(x_stream),)

            if self.is_gradient_checkpointing and self.training and (not use_cache) and (not output_attentions):
                # Checkpoint only the tensor output; caching/attn-weights are disabled in this path.
                x_stream = torch.utils.checkpoint.checkpoint(
                    lambda x: block(
                        x,
                        past_key_values=None,
                        cache_position=cache_position,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i],
                        use_cache=False,
                        output_attentions=False,
                    )[0],
                    x_stream,
                    use_reentrant=False,
                )
                attn_weights = None
            else:
                x_stream, past_key_values, attn_weights = block(
                    x_stream,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                all_self_attentions = all_self_attentions + (attn_weights,)

        hidden_states = self._readout(x_stream)
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs = (hidden_states, past_key_values, all_hidden_states, all_self_attentions)
            return tuple(o for o in outputs if o is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class MhcGPT2LMHeadModel(MhcGPT2PreTrainedModel, GenerationMixin):
    # Match HF GPT2LMHeadModel so safe serialization (safetensors) can handle tied weights.
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MhcGPT2Config):
        super().__init__(config)
        self.transformer = MhcGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        # Let PreTrainedModel handle weight tying (same behavior as HF GPT2LMHeadModel).
        return super().tie_weights()

    # Copied/adapted from GPT2LMHeadModel in Transformers 4.57.x
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = {}
        model_inputs["cache_position"] = cache_position

        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )

        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"

        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask = kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"

        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs[position_ids_key] = position_ids

        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        if (
            isinstance(past_key_values, Cache)
            and past_key_values.is_compileable
            and attention_mask is not None
            and attention_mask.ndim == 2
        ):
            if not self.config.is_encoder_decoder and model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]

            base_model = getattr(self, self.base_model_prefix, self)
            decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
            causal_mask_creation_function = getattr(
                base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
            if causal_mask_creation_function is None and decoder is not None:
                causal_mask_creation_function = getattr(
                    decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )

            if causal_mask_creation_function is None:
                token_type_ids = model_inputs.get("token_type_ids")
                position_ids = model_inputs.get(position_ids_key)
                causal_mask_creation_function = getattr(self, "create_masks_for_generate", create_masks_for_generate)
                attention_mask = causal_mask_creation_function(
                    config=self.config,
                    input_embeds=torch.empty((batch_size, sequence_length), dtype=self.dtype),
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    token_type_ids=token_type_ids,
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )

        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask

        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        model_inputs.pop("labels", None)
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = transformer_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + tuple(transformer_outputs[k] for k in ["past_key_values", "hidden_states", "attentions"])
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


