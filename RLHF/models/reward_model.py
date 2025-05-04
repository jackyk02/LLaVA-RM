# Copyright 2023 The LLaVA-RLHF Team
# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import Namespace
from dataclasses import dataclass
import os
from typing import Optional, Dict, Sequence, Union

import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import transformers
from transformers.trainer_utils import EvalPrediction
from transformers.utils.generic import ModelOutput

from peft import PeftModel, LoraModel, LoraConfig

from models.qlora_model import get_accelerate_model

from llava.model import *


def unpack_dict(
    d: Dict, keys: Sequence[str], return_type: type = tuple
) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def batch_select(input: Tensor, index: Tensor):
    """Select elements from a batched tensor with a batched index tensor.

    Example:
        input = torch.tensor([
            [0, 1, 2],
            [3, 0, 9],
            [6, 7, 8],
        ])
        index = torch.tensor([[0, 1], [1, 0], [0, 0]])
        batch_select(input, index) = tensor([
            [0, 1],
            [0, 3],
            [6, 6]
        ])
    """
    dummy_index = torch.arange(input.size(0), device=input.device).unsqueeze(-1)
    return input[dummy_index, index]


def make_generative_vlm(
    args: Namespace,
    model_name_or_path: str,
    qlora: bool = False,
    checkpoint_dir: Optional[str] = None,
    adapter_name="lora_default",
    is_trainable=True,
    reuse_base_model=False,
    tokenizer=None,
    **kwargs,
):
    if qlora:
        if checkpoint_dir is None or checkpoint_dir in ["scratch", "none"]:
            return get_accelerate_model(args, None, tokenizer=tokenizer)
        else:
            return get_accelerate_model(
                args,
                checkpoint_dir=checkpoint_dir,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
                reuse_base_model=reuse_base_model,
                tokenizer=tokenizer,
            )
    else:
        raise ValueError(f"Unknown model type: {model_name_or_path}")


def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, PeftModel):
        return get_transformer_hidden_size(model.base_model)

    if isinstance(model, LoraModel):
        return get_transformer_hidden_size(model.model)

    if isinstance(model, transformers.GPT2LMHeadModel):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    elif "modelling_RW.RWModel" in str(
        type(model)
    ) or "modelling_RW.RWForCausalLM" in str(type(model)):
        # TODO(zhiqings): Hack to add support for Falcon.
        hidden_size_attr_name = "hidden_size"
    else:
        # Hack to deal with the fact that transformers library changed the LLaMA model name.
        llama_cls = getattr(
            transformers,
            "LLaMAForCausalLM"
            if hasattr(transformers, "LLaMAForCausalLM")
            else "LlamaForCausalLM",
        )
        if isinstance(model, llama_cls) or "LlamaForCausalLM" in str(type(model)):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
        from typing import Any, Mapping
    return getattr(model.config, hidden_size_attr_name)


class RewardConfig(transformers.PretrainedConfig):
    model_type = "reward_model"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super(RewardConfig, self).__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path

@dataclass
class RewardModelOutput(ModelOutput):
    rewards: Tensor = None


class RewardModel(transformers.PreTrainedModel):
    config_class = RewardConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        args: Namespace,
        config: RewardConfig,
        checkpoint_dir: Optional[str] = None,
        adapter_name="lora_default",
        tokenizer=None,
        **kwargs,
    ):
        super(RewardModel, self).__init__(config)
        self.adapter_name = adapter_name
        self.backbone_model = make_generative_vlm(
            args,
            config.backbone_model_name_or_path,
            checkpoint_dir=checkpoint_dir,
            adapter_name=adapter_name,
            tokenizer=tokenizer,
            **kwargs,
        )
        hidden_size = get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(reward_head.bias)
        device = next(self.backbone_model.parameters()).device
        self.reward_head = reward_head.to(device)

        if checkpoint_dir is not None:
            reward_head_path = os.path.join(checkpoint_dir, "reward_head")
            if os.path.exists(reward_head_path):
                self.reward_head.load_state_dict(
                    torch.load(
                        reward_head_path,
                        map_location="cpu",
                    )
                )
            else:
                print(f"Warning: reward head not found at {reward_head_path}")

        self.reward_head.requires_grad_(kwargs.get("is_trainable", True))

    def forward(
        self, input_ids, attention_mask=None, images=None, return_dict=True, **kwargs
    ):
        # We only compute the rewards and don't compute the logistic regression loss in this function so that it's
        # easier to use for later stages of reranking / RL training.
        self.backbone_model.set_adapter(self.adapter_name)
        self.backbone_model.config.use_cache = False

        # print(input_ids.shape, images.shape, 'images', images.dtype)
        outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            images=images,
            **kwargs,
        )
        last_hidden_state = outputs.hidden_states[-1]
        assert isinstance(last_hidden_state, torch.Tensor), f"{outputs}"
        # last_hidden_state = outputs.last_hidden_state
        # TODO(zhiqings): Hacking to make sure every parameter is used in the backward pass.
        logits = outputs.logits
        last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits)

        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        # TODO(lxuechen): Make returning rewards at all positions and last_hidden_state an option.
        # last_hidden_state_at_the_end = last_hidden_state_at_the_end.type_as(
        #     next(self.reward_head.parameters()) # HACK(sheng): error with data parallel
        # )
        last_hidden_state_at_the_end = last_hidden_state_at_the_end.type_as(
            self.reward_head.weight
        )
        # print(last_hidden_state_at_the_end.device, self.reward_head.weight.device, self.reward_head.bias.device)
        rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, transformers.LlamaModel):
            module.gradient_checkpointing = value

        # TODO(zhiqings): Hack to add support for Falcon.
        if "RWModel" in str(type(module)):
            module.gradient_checkpointing = value


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class RewardModelTrainer(transformers.Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ["mm_projector", "embed_tokens", "embed_in"]
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split("/")[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )

        super(RewardModelTrainer, self)._save(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False, alpha=1):
        """
        Computes the loss for the reward model training.

        Args:
            model: The reward model.
            inputs (dict): Dictionary containing input tensors:
                - input_ids: (bsz, num_candidates, seq_len)
                - attention_mask: (bsz, num_candidates, seq_len)
                - index_0: (bsz, num_pairs) indices for the first candidate in pairs
                - index_1: (bsz, num_pairs) indices for the second candidate in pairs
                - choice: (bsz, num_pairs) 1 if candidate index_1 is preferred, 0 otherwise
                - images: (bsz, h, w, c) or similar image tensor
                - nrmse_0: (bsz, num_pairs) Ground truth metric (e.g., RMSE diff) for candidate index_0
                - nrmse_1: (bsz, num_pairs) Ground truth metric (e.g., RMSE diff) for candidate index_1
            return_outputs (bool): Whether to return intermediate outputs along with the loss.

        Returns:
            torch.Tensor or tuple: The computed loss, or a tuple (loss, outputs_dict) if return_outputs is True.
        """
        # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
        # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
        # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise.
        # nrmse_0, nrmse_1 are metrics associated with the candidates indicated by index_0 and index_1 respectively.
        input_ids, attention_mask, index_0, index_1, choice, images, nrmse_0, nrmse_1 = unpack_dict(
            inputs,
            keys=(
                "input_ids",
                "attention_mask",
                "index_0",
                "index_1",
                "choice",
                "images",
                "nrmse_0", # Shape: (bsz, num_pairs)
                "nrmse_1"  # Shape: (bsz, num_pairs)
            ),
        )
        # repeat images to match the number of candidates
        images = images.unsqueeze(1).repeat(1, input_ids.size(1), 1, 1, 1)
        images = einops.rearrange(images, "b n h w c -> (b n) h w c") # Adapt 'h w c' if needed

        num_candidates, num_pairs = input_ids.size(1), choice.size(1)
        input_ids_flat, attention_mask_flat = tuple(
            einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
        )
        # Assume model outputs a dictionary or object with a 'rewards' attribute
        outputs = model(
            input_ids=input_ids_flat, attention_mask=attention_mask_flat, images=images
        )
        # Assuming outputs.rewards exists and is the raw reward score
        rewards_flat = outputs.rewards
        rewards = einops.rearrange(
            rewards_flat, "(b c) -> b c", c=num_candidates
        )  # Size: (bsz, num_candidates).

        # Get rewards for the pairs being compared
        rewards_0, rewards_1 = tuple(
            batch_select(rewards, index) for index in (index_0, index_1)
        )  # Size: (bsz, num_pairs).

        # Original logits calculation: R(a_1) - R(a_0)
        logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).

        # --- Start Modification based on the formula ---

        # Ensure choice, nrmse_0, nrmse_1 are on the same device and dtype as logits
        logits_dtype = logits.dtype
        logits_device = logits.device
        choice_float = choice.to(dtype=logits_dtype, device=logits_device)
        nrmse_0 = nrmse_0.to(dtype=logits_dtype, device=logits_device)
        nrmse_1 = nrmse_1.to(dtype=logits_dtype, device=logits_device)

        # Identify rewards for the winning (W) and losing (L) actions based on 'choice'
        # reward_W = R_phi(a_t^W, s_t, I)
        # reward_L = R_phi(a_t^L, s_t, I)
        # If choice=1, W is index_1, L is index_0. reward_W=rewards_1, reward_L=rewards_0
        # If choice=0, W is index_0, L is index_1. reward_W=rewards_0, reward_L=rewards_1
        reward_W = choice_float * rewards_1 + (1 - choice_float) * rewards_0
        reward_L = choice_float * rewards_0 + (1 - choice_float) * rewards_1

        # Calculate the predicted reward difference for the chosen pair: R(a_W) - R(a_L)
        # This should always be positive or zero if the model aligns with 'choice'.
        # Can be calculated as: (2 * choice_float - 1) * logits
        predicted_reward_diff = reward_W - reward_L # Equivalent to (2 * choice_float - 1) * logits

        # Calculate predicted preference level: delta_hat = |R(a_W) - R(a_L)|
        # Note: Based on the formula, it seems delta_hat uses the absolute difference
        # between the *actual* computed rewards R(a_W) and R(a_L), which is |predicted_reward_diff|.
        # delta_hat = torch.abs(predicted_reward_diff)
        # However, the text defines delta_hat as |R(aW) - R(aL)| where aW and aL are the specific
        # actions chosen. This implies using the absolute value of the difference calculated above.
        delta_hat = torch.abs(predicted_reward_diff)
        # An alternative interpretation could be delta_hat = |rewards_1 - rewards_0| = |logits|,
        # if the formula meant the absolute difference between *any* pair's rewards.
        # Let's stick to the definition based on R(aW) and R(aL).
        # delta_hat = torch.abs(reward_W - reward_L) # which is predicted_reward_diff.abs()

        # Identify the ground truth metrics for the winning (W) and losing (L) actions
        # rmse_W = RMSE(a_t^W, a_t^*)
        # rmse_L = RMSE(a_t^L, a_t^*)
        # If choice=1, W is index_1, L is index_0. rmse_W=nrmse_1, rmse_L=nrmse_0
        # If choice=0, W is index_0, L is index_1. rmse_W=nrmse_0, rmse_L=nrmse_1
        rmse_W = choice_float * nrmse_1 + (1 - choice_float) * nrmse_0
        rmse_L = choice_float * nrmse_0 + (1 - choice_float) * nrmse_1

        # Calculate ground truth preference level: delta_star = |RMSE(a_W) - RMSE(a_L)|
        delta_star = torch.abs(rmse_W - rmse_L)
        # Note: This simplifies to torch.abs(nrmse_1 - nrmse_0)

        # Define the hyperparameter alpha (should be configured appropriately)

        # Calculate the penalty term: alpha * || delta_star - delta_hat ||^2_2
        # Note: ||x||^2_2 is just x^2 for scalars.
        preference_level_penalty = alpha * (delta_star - delta_hat)**2

        # Calculate the argument for the sigmoid function in the loss formula:
        # R(a_W) - R(a_L) - alpha * || delta_star - delta_hat ||^2_2
        modified_reward_diff = predicted_reward_diff - preference_level_penalty

        # Calculate the final loss using log sigmoid for numerical stability:
        # Loss = -E[ log(sigma(modified_reward_diff)) ]
        # This is equivalent to binary cross-entropy with logits where the target is always 1.
        loss = -F.logsigmoid(modified_reward_diff).mean()

        # --- End Modification ---

        # Original regularization term (commented out as it's not in the provided formula)
        # loss = loss + (rewards_1 + rewards_0).mean().abs() * 1e-3

        # Log the original paired rewards for monitoring/debugging if needed
        logged_rewards = torch.stack((rewards_1, rewards_0), dim=-1)

        # Return loss, and optionally intermediate values for logging
        return (loss, dict(logits=logged_rewards)) if return_outputs else loss


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
    #     # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
    #     # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise.
    #     input_ids, attention_mask, index_0, index_1, choice, images, nrmse_0, nrmse_1 = unpack_dict(
    #         inputs,
    #         keys=(
    #             "input_ids",
    #             "attention_mask",
    #             "index_0",
    #             "index_1",
    #             "choice",
    #             "images",
    #             "nrmse_0",
    #             "nrmse_1"
    #         ),
    #     )
    #     # repeat images to match the number of candidates
    #     images = images.unsqueeze(1).repeat(1, input_ids.size(1), 1, 1, 1)
    #     images = einops.rearrange(images, "b n h w c -> (b n) h w c")

    #     num_candidates, num_pairs = input_ids.size(1), choice.size(1)
    #     input_ids_flat, attention_mask_flat = tuple(
    #         einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
    #     )
    #     outputs = model(
    #         input_ids=input_ids_flat, attention_mask=attention_mask_flat, images=images
    #     )
    #     rewards_flat = outputs.rewards
    #     rewards = einops.rearrange(
    #         rewards_flat, "(b c) -> b c", c=num_candidates
    #     )  # Size: (bsz, num_candidates).

    #     rewards_0, rewards_1 = tuple(
    #         batch_select(rewards, index) for index in (index_0, index_1)
    #     )  # Size: (bsz, num_pairs).
    #     logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).
    #     # Type casting of `choice` is due to amp.autocast context manager.
    #     loss = F.binary_cross_entropy_with_logits(
    #         logits, choice.to(logits.dtype), reduction="mean"
    #     )

    #     loss = loss + (rewards_1 + rewards_0).mean().abs() * 1e-3

    #     logged_rewards = torch.stack((rewards_1, rewards_0), dim=-1)
    #     return (loss, dict(logits=logged_rewards)) if return_outputs else loss

    # # bce + mse
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
    #     # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
    #     # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise.
    #     input_ids, attention_mask, index_0, index_1, choice, images, nrmse_0, nrmse_1 = unpack_dict(
    #         inputs,
    #         keys=(
    #             "input_ids",
    #             "attention_mask",
    #             "index_0",
    #             "index_1",
    #             "choice",
    #             "images",
    #             "nrmse_0",
    #             "nrmse_1"
    #         ),
    #     )

    #     # repeat images to match the number of candidates
    #     images = images.unsqueeze(1).repeat(1, input_ids.size(1), 1, 1, 1)
    #     images = einops.rearrange(images, "b n h w c -> (b n) h w c")

    #     num_candidates, num_pairs = input_ids.size(1), choice.size(1)
    #     input_ids_flat, attention_mask_flat = tuple(
    #         einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
    #     )
    #     outputs = model(
    #         input_ids=input_ids_flat, attention_mask=attention_mask_flat, images=images
    #     )
    #     rewards_flat = outputs.rewards
    #     rewards = einops.rearrange(
    #         rewards_flat, "(b c) -> b c", c=num_candidates
    #     )  # Size: (bsz, num_candidates).

    #     rewards_0, rewards_1 = tuple(
    #         batch_select(rewards, index) for index in (index_0, index_1)
    #     )  # Size: (bsz, num_pairs).
    #     logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).

    #     # Prepare NRMSE values
    #     nrmse_0 = nrmse_0.view(nrmse_0.size(0), 1)  # reshape to [batch_size, 1]
    #     nrmse_1 = nrmse_1.view(nrmse_1.size(0), 1)  # reshape to [batch_size, 1]

    #     # Calculate target difference for MSE loss
    #     target_diff = nrmse_0 - nrmse_1  # If nrmse_1 < nrmse_0 => target_diff > 0
    #     target_diff = target_diff.to(logits.dtype)

    #     # Calculate losses
    #     bce_loss = F.binary_cross_entropy_with_logits(
    #         logits,
    #         choice.to(logits.dtype),
    #         reduction="mean"
    #     )
        
    #     eps = 1e-8
    #     target_diff = (target_diff - target_diff.min()) / (target_diff.max() - target_diff.min() + eps)

    #     # MSE loss between logits and NRMSE difference
    #     mse_loss = F.mse_loss(torch.sigmoid(logits), target_diff)
        
    #     # Combine losses with alpha parameter
    #     alpha = 0.5  # hyperparameter to tune

    #     # Add regularization term
    #     regularization = (rewards_1 + rewards_0).mean().abs() * 1e-3
    #     loss = bce_loss + alpha * mse_loss + regularization
        
    #     logged_rewards = torch.stack((rewards_1, rewards_0), dim=-1)
    #     return (loss, dict(logits=[logged_rewards])) if return_outputs else loss

def compute_reward_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
    # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
    logits = torch.tensor(
        eval_prediction.predictions[..., 0] - eval_prediction.predictions[..., 1]
    ).squeeze(-1)
    labels = torch.tensor(eval_prediction.label_ids[-1]).squeeze(-1)
    predictions = (logits >= 0.0).long()
    
    # Calculate original metrics
    accuracy = predictions.eq(labels).float().mean().item()
    label_positive_rate = (labels == 1).float().mean().item()
    average_score = torch.tensor(eval_prediction.predictions).float().mean().item()
    
    # Calculate components needed for F1 score
    true_positives = ((predictions == 1) & (labels == 1)).float().sum().item()
    false_positives = ((predictions == 1) & (labels == 0)).float().sum().item()
    false_negatives = ((predictions == 0) & (labels == 1)).float().sum().item()
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return dict(
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
        label_positive_rate=label_positive_rate,
        average_score=average_score,
    )

# def compute_reward_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
#     # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
#     logits = torch.tensor(
#         eval_prediction.predictions[..., 0] - eval_prediction.predictions[..., 1]
#     ).squeeze(-1)
#     labels = torch.tensor(eval_prediction.label_ids[-1]).squeeze(-1)
#     predictions = (logits >= 0.0).long()
#     accuracy = predictions.eq(labels).float().mean().item()
#     label_positive_rate = (labels == 1).float().mean().item()
#     average_score = torch.tensor(eval_prediction.predictions).float().mean().item()
#     return dict(
#         accuracy=accuracy,
#         label_positive_rate=label_positive_rate,
#         average_score=average_score,
#     )


def load_4bit_reward_model_for_inference(
    checkpoint_dir: str,
    vision_tower: str = None,
    lora_modules: list = None,
    image_aspect_ratio: str = "square",
    image_grid_pinpoints: int = None,
    bits: int = 4,
    fp16: bool = False,
    bf16: bool = False,
    double_quant: bool = True,
    quant_type: str = "nf4",
    gradient_checkpointing: bool = False,
    adapter_name="lora_default",
    is_trainable=True,
    reuse_base_model=False,
    trust_remote_code=False,
):
    # Load the model.
    lora_checkpoint_dir = checkpoint_dir
    if os.path.exists(os.path.join(lora_checkpoint_dir, "adapter_model")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "adapter_model")
    if os.path.exists(os.path.join(lora_checkpoint_dir, "lora_default")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "lora_default")

    lora_config = LoraConfig.from_pretrained(lora_checkpoint_dir)
    config = RewardConfig(
        backbone_model_name_or_path=lora_config.base_model_name_or_path
    )

    args = Namespace(
        model_name_or_path=config.backbone_model_name_or_path,
        vision_tower=vision_tower,
        lora_modules=lora_modules,
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints,
        bits=bits,
        fp16=fp16,
        bf16=bf16,
        double_quant=double_quant,
        quant_type=quant_type,
        trust_remote_code=trust_remote_code,
        full_finetune=False,
        gradient_checkpointing=gradient_checkpointing,
    )

    model = RewardModel(
        args,
        config,
        checkpoint_dir=checkpoint_dir,
        qlora=bits == 4 or bits == 8,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
        reuse_base_model=reuse_base_model,
    )
    return model
