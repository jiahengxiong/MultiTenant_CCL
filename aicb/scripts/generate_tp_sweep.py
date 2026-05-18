import argparse
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.utils import get_params
from workload_generator.generate_megatron_workload import MegatronWorkload
from workload_generator.mocked_model.training.MockedDeepSeek import DeepSeekV3Model
from workload_generator.mocked_model.training.MockedMegatron import MegatronModel


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _lcm(a, b):
    return a // _gcd(a, b) * b


def _ceil_to_multiple(x, m):
    return ((x + m - 1) // m) * m


def _run_one(argv, out_path):
    if os.path.exists(out_path):
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    orig_argv = sys.argv
    try:
        sys.argv = ["generate_tp_sweep.py"] + argv
        args = get_params()
        if args.frame == "DeepSeek":
            model = DeepSeekV3Model(args)
        else:
            model = MegatronModel(args)
        workload_generator = MegatronWorkload(args, model)
        workload = workload_generator()
        tmp_base = f"sweep_{args.model_name}_ws{args.world_size}"
        workload.dump(tmp_base)
        tmp_csv = os.path.join(
            "results",
            "mocked_workload",
            f"{tmp_base}_workload.csv",
        )
        shutil.move(tmp_csv, out_path)
    finally:
        sys.argv = orig_argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=1024)
    parser.add_argument("--seq_length", type=int, default=2048)
    args = parser.parse_args()

    base_dir = os.path.join("results", "mocked_workload_sweep")

    for world_size in range(args.start, args.end + 1):
        tp = world_size

        gpt_heads = 40
        gpt_hidden = _ceil_to_multiple(5120, _lcm(tp, gpt_heads))

        llama_heads = 64
        llama_hidden = _ceil_to_multiple(8192, _lcm(tp, llama_heads))
        llama_ffn = int((4 * llama_hidden * 2 / 3) / 64) * 64
        llama_ffn = _ceil_to_multiple(llama_ffn, tp)

        deepseek_heads = _ceil_to_multiple(16, tp)
        deepseek_hidden = _ceil_to_multiple(10944, _lcm(tp, deepseek_heads))
        deepseek_ffn = _ceil_to_multiple(1408, tp)

        _run_one(
            [
                "--frame",
                "Megatron",
                "--model_name",
                "gpt_13B",
                "--world_size",
                str(world_size),
                "--tensor_model_parallel_size",
                str(tp),
                "--pipeline_model_parallel",
                "1",
                "--global_batch",
                "1",
                "--micro_batch",
                "1",
                "--epoch_num",
                "1",
                "--num_layers",
                "40",
                "--hidden_size",
                str(gpt_hidden),
                "--num_attention_heads",
                str(gpt_heads),
                "--seq_length",
                str(args.seq_length),
                "--vocab_size",
                "50257",
                "--max_position_embeddings",
                str(args.seq_length),
                "--use-distributed-optimizer",
                "--workload_only",
            ],
            os.path.join(base_dir, "gpt_13B", f"world_size{world_size}.csv"),
        )

        _run_one(
            [
                "--frame",
                "Megatron",
                "--model_name",
                "llama_65B",
                "--world_size",
                str(world_size),
                "--tensor_model_parallel_size",
                str(tp),
                "--pipeline_model_parallel",
                "1",
                "--global_batch",
                "1",
                "--micro_batch",
                "1",
                "--epoch_num",
                "1",
                "--num_layers",
                "80",
                "--hidden_size",
                str(llama_hidden),
                "--ffn_hidden_size",
                str(llama_ffn),
                "--num_attention_heads",
                str(llama_heads),
                "--seq_length",
                str(args.seq_length),
                "--vocab_size",
                "32000",
                "--max_position_embeddings",
                str(args.seq_length),
                "--use_flash_attn",
                "--swiglu",
                "--use-distributed-optimizer",
                "--workload_only",
            ],
            os.path.join(base_dir, "llama_65B", f"world_size{world_size}.csv"),
        )

        _run_one(
            [
                "--frame",
                "DeepSeek",
                "--model_name",
                "DeepSeek_16B",
                "--world_size",
                str(world_size),
                "--tensor_model_parallel_size",
                str(tp),
                "--pipeline_model_parallel",
                "1",
                "--global_batch",
                "1",
                "--micro_batch",
                "1",
                "--epoch_num",
                "1",
                "--num_layers",
                "27",
                "--hidden_size",
                str(deepseek_hidden),
                "--ffn_hidden_size",
                str(deepseek_ffn),
                "--num_attention_heads",
                str(deepseek_heads),
                "--seq_length",
                str(args.seq_length),
                "--vocab_size",
                "32000",
                "--max_position_embeddings",
                str(args.seq_length),
                "--use-distributed-optimizer",
                "--moe_enable",
                "--num_experts",
                "64",
                "--moe_router_topk",
                "2",
                "--expert_model_parallel_size",
                "1",
                "--enable_sequence_parallel",
                "--n_shared_expert",
                "2",
                "--n_dense_layers",
                "1",
                "--q_lora_rank",
                "0",
                "--kv_lora_rank",
                "512",
                "--qk_nope_dim",
                "128",
                "--qk_rope_dim",
                "64",
                "--v_head_dim",
                "128",
                "--workload_only",
            ],
            os.path.join(base_dir, "DeepSeek_16B", f"world_size{world_size}.csv"),
        )


if __name__ == "__main__":
    main()
