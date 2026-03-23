from peft import (
    set_peft_model_state_dict,
    get_peft_model_state_dict
)
import torch
import os
from torch.nn.functional import normalize


def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir)
        if k == 0:
            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                       single_weights.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}

    set_peft_model_state_dict(model, weighted_single_weights, "default")

    return model

def FedSA(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):

    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    aggregated_A = {}

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(
            output_dir, str(epoch), "local_output_{}".format(client_id), "pytorch_model.bin")
        
        single_weights = torch.load(single_output_dir)

        A_state_dict = {key: value.clone() for key, value in single_weights.items() if 'lora_A' in key}

        # Averaging
        for key in A_state_dict.keys():
            A_state_dict[key] *= weights_array[k]

        # client 0
        if k == 0:
            # Initialization
            aggregated_A = A_state_dict
        else:
            # Add the weighted LoRA A matrices from subsequent clients to the aggregated_A (Averaged)
            for key in aggregated_A.keys():
                aggregated_A[key] += A_state_dict[key]

    # global_peft_state_dict = model.get_peft_model_state_dict()
    global_peft_state_dict = get_peft_model_state_dict(model, adapter_name="default")

# 
    for key in aggregated_A.keys():
        global_peft_state_dict[key] = aggregated_A[key]


    set_peft_model_state_dict(model, global_peft_state_dict, adapter_name="default")

    return model


def SCAFFOLD(
    model,
    selected_clients_set,
    output_dir,
    local_dataset_len_dict,
    epoch,
    scaffold_state=None,
    num_clients=None,
    local_steps=1,
    local_lr=1.0,
):
    """
    Server-side SCAFFOLD aggregation for LoRA adapter weights.

    Args:
        model: Global PEFT model.
        selected_clients_set: Clients selected in the current communication round.
        output_dir: Root output dir that stores local client adapters.
        local_dataset_len_dict: Mapping from client id to local dataset size.
        epoch: Current communication round.
        scaffold_state: Persistent state dict across rounds. Expected keys:
            - "server_control": global control variate state_dict
            - "client_controls": per-client control variate state_dicts
        num_clients: Total number of clients. Needed for the server control update.
        local_steps: Number of local optimizer steps used by each selected client.
            Can be a scalar or a dict keyed by client_id.
        local_lr: Local learning rate used by each selected client.

    Returns:
        (model, scaffold_state)

    Notes:
        This function implements the server-side control variate update and keeps
        all state in a single dict for easy use in main(). A mathematically
        complete SCAFFOLD pipeline also needs the client-side local optimizer to
        apply the correction term (c - c_i) during local training.
    """
    if num_clients is None:
        raise ValueError("SCAFFOLD requires num_clients to update the server control variate.")
    if local_lr == 0:
        raise ValueError("SCAFFOLD requires local_lr != 0.")

    selected_clients = list(selected_clients_set)
    weights_array = normalize(
        torch.tensor(
            [local_dataset_len_dict[client_id] for client_id in selected_clients],
            dtype=torch.float32,
        ),
        p=1,
        dim=0,
    )

    global_peft_state_dict = get_peft_model_state_dict(model, adapter_name="default")
    global_state_cpu = {key: value.detach().cpu().clone() for key, value in global_peft_state_dict.items()}

    if scaffold_state is None:
        scaffold_state = {}

    server_control = scaffold_state.get("server_control")
    if server_control is None:
        server_control = {
            key: torch.zeros_like(value, device="cpu")
            for key, value in global_state_cpu.items()
        }
    else:
        server_control = {
            key: server_control[key].detach().cpu().clone()
            for key in global_state_cpu.keys()
        }

    client_controls = scaffold_state.get("client_controls", {})

    aggregated_weights = None
    weighted_delta_c_sum = None
    for k, client_id in enumerate(selected_clients):
        single_output_dir = os.path.join(
            output_dir,
            str(epoch),
            "local_output_{}".format(client_id),
            "pytorch_model.bin",
        )
        single_weights = torch.load(single_output_dir, map_location="cpu")
        client_local_steps = local_steps[client_id] if isinstance(local_steps, dict) else local_steps
        if client_local_steps <= 0:
            raise ValueError(f"SCAFFOLD requires local_steps > 0 for client {client_id}.")
        step_scale = float(client_local_steps) * float(local_lr)

        if aggregated_weights is None:
            aggregated_weights = {
                key: single_weights[key].detach().clone() * weights_array[k]
                for key in single_weights.keys()
            }
        else:
            for key in aggregated_weights.keys():
                aggregated_weights[key] += single_weights[key].detach() * weights_array[k]

        client_control = client_controls.get(client_id)
        if client_control is None:
            client_control = {
                key: torch.zeros_like(value, device="cpu")
                for key, value in global_state_cpu.items()
            }
        else:
            client_control = {
                key: client_control[key].detach().cpu().clone()
                for key in global_state_cpu.keys()
            }

        updated_client_control = {}
        delta_client_control = {}
        for key in global_state_cpu.keys():
            updated_client_control[key] = (
                client_control[key]
                - server_control[key]
                + (global_state_cpu[key] - single_weights[key].detach().cpu()) / step_scale
            )
            delta_client_control[key] = updated_client_control[key] - client_control[key]

        client_controls[client_id] = updated_client_control

        if weighted_delta_c_sum is None:
            weighted_delta_c_sum = {
                key: delta_client_control[key].clone() * weights_array[k]
                for key in delta_client_control.keys()
            }
        else:
            for key in weighted_delta_c_sum.keys():
                weighted_delta_c_sum[key] += delta_client_control[key] * weights_array[k]

    updated_server_control = {}
    for key in server_control.keys():
        updated_server_control[key] = server_control[key] + weighted_delta_c_sum[key]

    set_peft_model_state_dict(model, aggregated_weights, "default")

    scaffold_state["server_control"] = updated_server_control
    scaffold_state["client_controls"] = client_controls
    return model, scaffold_state


def HAA(
    model,
    selected_clients_set,
    output_dir,
    local_dataset_len_dict,
    epoch,
    scaffold_state=None,
    num_clients=None,
    local_steps=1,
    local_lr=1.0,
    tau=5.0,
    eps=1e-12,
):
    """
    SCAFFOLD-based server aggregation with heterogeneity-adaptive weights.

    Similarity:
        s_k = cos(Delta_phi_k, Delta)
    Adaptive weight:
        alpha_k = n_k * exp(tau * s_k) / sum_i n_i * exp(tau * s_i)
    Aggregated update:
        Delta_het = sum_k alpha_k * Delta_phi_k
    """
    if num_clients is None:
        raise ValueError("HAA requires num_clients to update the server control variate.")
    if local_lr == 0:
        raise ValueError("HAA requires local_lr != 0.")

    selected_clients = list(selected_clients_set)
    if len(selected_clients) == 0:
        return model, scaffold_state if scaffold_state is not None else {}

    global_peft_state_dict = get_peft_model_state_dict(model, adapter_name="default")
    global_state_cpu = {key: value.detach().cpu().clone() for key, value in global_peft_state_dict.items()}

    if scaffold_state is None:
        scaffold_state = {}

    server_control = scaffold_state.get("server_control")
    if server_control is None:
        server_control = {
            key: torch.zeros_like(value, device="cpu")
            for key, value in global_state_cpu.items()
        }
    else:
        server_control = {
            key: server_control[key].detach().cpu().clone()
            for key in global_state_cpu.keys()
        }

    client_controls = scaffold_state.get("client_controls", {})

    nk_array = torch.tensor(
        [local_dataset_len_dict[client_id] for client_id in selected_clients],
        dtype=torch.float32,
    )
    base_weights = normalize(nk_array, p=1, dim=0)

    client_payloads = []
    for client_id in selected_clients:
        single_output_dir = os.path.join(
            output_dir,
            str(epoch),
            "local_output_{}".format(client_id),
            "pytorch_model.bin",
        )
        single_weights = torch.load(single_output_dir, map_location="cpu")
        client_local_steps = local_steps[client_id] if isinstance(local_steps, dict) else local_steps
        if client_local_steps <= 0:
            raise ValueError(f"HAA requires local_steps > 0 for client {client_id}.")
        step_scale = float(client_local_steps) * float(local_lr)

        delta_weights = {
            key: single_weights[key].detach().cpu() - global_state_cpu[key]
            for key in global_state_cpu.keys()
        }

        client_control = client_controls.get(client_id)
        if client_control is None:
            client_control = {
                key: torch.zeros_like(value, device="cpu")
                for key, value in global_state_cpu.items()
            }
        else:
            client_control = {
                key: client_control[key].detach().cpu().clone()
                for key in global_state_cpu.keys()
            }

        updated_client_control = {}
        delta_client_control = {}
        for key in global_state_cpu.keys():
            updated_client_control[key] = (
                client_control[key]
                - server_control[key]
                + (global_state_cpu[key] - single_weights[key].detach().cpu()) / step_scale
            )
            delta_client_control[key] = updated_client_control[key] - client_control[key]

        client_controls[client_id] = updated_client_control
        client_payloads.append(
            {
                "client_id": client_id,
                "n_k": float(local_dataset_len_dict[client_id]),
                "delta_weights": delta_weights,
                "delta_client_control": delta_client_control,
            }
        )

    consensus_delta = {key: torch.zeros_like(value, device="cpu") for key, value in global_state_cpu.items()}
    for k, payload in enumerate(client_payloads):
        for key in consensus_delta.keys():
            consensus_delta[key] += payload["delta_weights"][key] * base_weights[k]

    consensus_norm_sq = torch.tensor(0.0, dtype=torch.float32)
    for key in consensus_delta.keys():
        consensus_norm_sq += torch.sum(consensus_delta[key] * consensus_delta[key]).float()
    consensus_norm = torch.sqrt(consensus_norm_sq + eps)

    similarities = []
    for payload in client_payloads:
        dot = torch.tensor(0.0, dtype=torch.float32)
        delta_norm_sq = torch.tensor(0.0, dtype=torch.float32)
        for key in consensus_delta.keys():
            delta_tensor = payload["delta_weights"][key]
            consensus_tensor = consensus_delta[key]
            dot += torch.sum(delta_tensor * consensus_tensor).float()
            delta_norm_sq += torch.sum(delta_tensor * delta_tensor).float()
        delta_norm = torch.sqrt(delta_norm_sq + eps)
        similarities.append((dot / (delta_norm * consensus_norm + eps)).clamp(min=-1.0, max=1.0))

    similarity_tensor = torch.stack(similarities)
    n_tensor = torch.tensor([payload["n_k"] for payload in client_payloads], dtype=torch.float32)
    if torch.any(n_tensor <= 0):
        raise ValueError("HAA requires each selected client to have local dataset size > 0.")

    # Stable form of n_k * exp(tau * s_k): exp(log(n_k) + tau * s_k)
    log_alpha = torch.log(n_tensor + eps) + float(tau) * similarity_tensor
    log_alpha = log_alpha - torch.max(log_alpha)
    alpha_unnormalized = torch.exp(log_alpha)
    alpha_weights = alpha_unnormalized / (alpha_unnormalized.sum() + eps)

    aggregated_weights = {key: value.clone() for key, value in global_state_cpu.items()}
    weighted_delta_c_sum = {key: torch.zeros_like(value, device="cpu") for key, value in server_control.items()}
    for k, payload in enumerate(client_payloads):
        alpha_k = alpha_weights[k]
        for key in aggregated_weights.keys():
            aggregated_weights[key] += payload["delta_weights"][key] * alpha_k
            weighted_delta_c_sum[key] += payload["delta_client_control"][key] * alpha_k

    updated_server_control = {}
    for key in server_control.keys():
        updated_server_control[key] = server_control[key] + weighted_delta_c_sum[key]

    set_peft_model_state_dict(model, aggregated_weights, "default")

    scaffold_state["server_control"] = updated_server_control
    scaffold_state["client_controls"] = client_controls
    scaffold_state["last_haa_similarity"] = {
        str(payload["client_id"]): float(similarity_tensor[k].item())
        for k, payload in enumerate(client_payloads)
    }
    scaffold_state["last_haa_alpha"] = {
        str(payload["client_id"]): float(alpha_weights[k].item())
        for k, payload in enumerate(client_payloads)
    }
    return model, scaffold_state
