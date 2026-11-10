import copy
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *


## ------------------------------------------------------ ##
model_str = 'babylm/babyllama-100m-2024'
base_model = AutoModelForCausalLM.from_pretrained(model_str)
tokenizer = AutoTokenizer.from_pretrained(model_str)

tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

## ------------------------------------------------------ ##
print(base_model)

## ------------------------------------------------------ ##
prompt = "The quick brown fox jumped over the "

input_ids = tokenizer(prompt, return_tensors= "pt")
print(input_ids)

with torch.no_grad():
    outputs = base_model.generate(**input_ids,
                                    max_new_tokens= 2,
                                    pad_token_id= tokenizer.pad_token_id)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens= True)
generated_portion = generated_text[len(prompt) : ]
print(f"Generated text: {prompt}\033[94m{generated_portion}\033[0m")

## ------------------------------------------------------ ##
ref_model = copy.deepcopy(base_model)

lora_config = LoraConfig(r= 8,
                        lora_alpha= 32,
                        target_modules= ["q_proj", "v_proj"],
                        lora_dropout= 0.1,
                        init_lora_weights= False,
                        bias= "none",
                        task_type= "CAUSAL_LM")

model = get_peft_model(base_model, lora_config)

## ------------------------------------------------------ ##
print(model)

## ------------------------------------------------------ ##
def prepare_inputs(prompt, completion):
    prompt_tokens = tokenizer(prompt, return_tensors= "pt")
    completion_tokens = tokenizer(completion, return_tensors= "pt")

    input_ids = torch.cat([prompt_tokens["input_ids"], completion_tokens["input_ids"]], dim= 1)

    attention_mask = torch.cat([prompt_tokens["attention_mask"],
                                completion_tokens["attention_mask"]],
                                dim= 1)

    prompt_length = prompt_tokens["input_ids"].shape[1]
    completion_length = completion_tokens["input_ids"].shape[1]
    total_length = prompt_length + completion_length

    completion_mask = torch.zeros(total_length, dtype= torch.float32)
    completion_mask[prompt_length : ] = 1.0

    return input_ids, attention_mask, completion_mask

## ------------------------------------------------------ ##
def compute_log_probs(model, input_ids, attention_mask):
    outputs = model(input_ids, attention_mask= attention_mask)

    log_probs = F.log_softmax(outputs.logits, dim= -1)

    return log_probs.gather(dim= -1, index= input_ids.unsqueeze(-1)).squeeze(-1)

## ------------------------------------------------------ ##
def grpo_loss(model, ref_model, prompt, completion, advantage):
    input_ids, attention_mask, completion_mask = prepare_inputs(prompt, completion)

    token_log_probs = compute_log_probs(model, input_ids, attention_mask)

    with torch.no_grad():
        ref_token_log_probs = compute_log_probs(ref_model, input_ids, attention_mask)

    # ratio = p_model / p_ref = exp(log(p_model) - log(p_ref))
    ratio = torch.exp(token_log_probs - ref_token_log_probs)

    policy_loss = ratio * advantage

    per_token_loss = -policy_loss

    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

    return loss

## ------------------------------------------------------ ##
grpo_loss(model, ref_model, prompt, "fence and", advantage= 2.0)

## ------------------------------------------------------ ##
grpo_loss(ref_model, ref_model, prompt, "fence and", advantage= 2.0)

## ------------------------------------------------------ ##
completion = "fence and"

input_ids, attention_mask, completion_mask = prepare_inputs(prompt, completion)

with torch.no_grad():
    token_log_probs = compute_log_probs(model, input_ids, attention_mask)

    ref_token_log_probs = compute_log_probs(ref_model, input_ids, attention_mask)

ratio = torch.exp(token_log_probs - ref_token_log_probs)
print(ratio)

## ------------------------------------------------------ ##
def grpo_loss_with_clip(model, ref_model, prompt, completion, advantage, epsilon= 0.2):
    input_ids, attention_mask, completion_mask = prepare_inputs(prompt, completion)

    token_log_probs = compute_log_probs(model, input_ids, attention_mask)

    with torch.no_grad():
        ref_token_log_probs = compute_log_probs(ref_model, input_ids, attention_mask)

    # ratio = p_model / p_ref = exp(log(p_model) - log(p_ref))
    ratio = torch.exp(token_log_probs - ref_token_log_probs)

    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage

    policy_loss = torch.min(unclipped, clipped)

    per_token_loss = -policy_loss

    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

    return loss

## ------------------------------------------------------ ##
grpo_loss_with_clip(model, ref_model, prompt, "fence and", advantage= 2.0, epsilon= 0.2)

## ------------------------------------------------------ ##
grpo_loss_with_clip(ref_model, ref_model, prompt, "fence and", advantage= 2.0)

## ------------------------------------------------------ ##
completion = "fence and"

input_ids, attention_mask, _ = prepare_inputs(prompt, completion)

with torch.no_grad():
    token_log_probs = compute_log_probs(model, input_ids, attention_mask)
    ref_token_log_probs = compute_log_probs(ref_model, input_ids, attention_mask)

with torch.no_grad():
    epsilon = 0.2
    ratio = torch.exp(token_log_probs - ref_token_log_probs)
    ratio_unclipped = ratio
    ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

visualize_clipped_ratios(ratio_unclipped[0][9 : ], ratio_clipped[0][9 : ], epsilon)

## ------------------------------------------------------ ##
def grpo_loss_with_kl(model, ref_model, prompt, completion, advantage, epsilon= 0.2, beta= 0.1):
    input_ids, attention_mask, completion_mask = prepare_inputs(prompt, completion)

    token_log_probs = compute_log_probs(model, input_ids, attention_mask)

    with torch.no_grad():
        ref_token_log_probs = compute_log_probs(ref_model, input_ids, attention_mask)

    # ratio = p_model / p_ref = exp(log(p_model) - log(p_ref))
    ratio = torch.exp(token_log_probs - ref_token_log_probs)

    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage

    policy_loss = torch.min(unclipped, clipped)

    delta = token_log_probs - ref_token_log_probs
    per_token_kl = torch.exp(-delta) + delta - 1

    per_token_loss = -(policy_loss - beta * per_token_kl)

    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

    return loss

## ------------------------------------------------------ ##
delta = np.linspace(-6, 6, 500)

kl_divergence = np.exp(-delta) + delta - 1

plt.figure(figsize=(8, 5))
plt.plot(delta, kl_divergence, label= r'$KL(\pi_{\mathrm{ref}} || \pi) = e^{-\Delta} + \Delta - 1$')
plt.axhline(0, color= 'gray', linestyle= '--', linewidth= 0.5)
plt.axvline(0, color= 'gray', linestyle= '--', linewidth= 0.5)
plt.fill_between(delta, kl_divergence, where= (delta > 0), color= 'red', alpha= 0.3,
                label= 'Overconfident region (Δ > 0)')
plt.fill_between(delta, kl_divergence, where= (delta < 0), color= 'green', alpha= 0.3,
                 label= 'Conservative region (Δ < 0)')
plt.title("KL Divergence as 'Gravitational Pull' Toward Reference Policy")
plt.xlabel(r'$\Delta = \log \pi - \log \pi_{\mathrm{ref}}$')
plt.ylabel('KL Penalty')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## ------------------------------------------------------ ##
for beta in [0, 0.1, 0.5]:
    loss = grpo_loss_with_kl(model,
                            ref_model,
                            prompt,
                            "fence and",
                            advantage= 2.0,
                            epsilon= 0.2,
                            beta= beta
                            )

    print(f"beta = {beta}")
    print(f"loss = {loss.item():.3f}")
    print()
