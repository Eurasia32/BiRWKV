#
# test_birwkv.py
#
# A script to test a pre-trained BiRWKV-LLaDA model and generate text
# using iterative denoising, as described in the LLaDA paper.
# This script imports the model definition from `train_birwkv.py`.
#
# 运行要求:
# 1. 确保 train_birwkv.py, birwkv_op.cpp, birwkv_kernel.cu 在同一目录。
# 2. 已安装必要的库 (torch, transformers, etc.)。
#
# 运行命令示例:
# python test_birwkv.py \
#   --checkpoint_path ./birwkv-model/checkpoint-10000/pytorch_model.bin \
#   --prompt "Once upon a time, in a land far, far away,"
#

import argparse
import logging
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# --- 从训练脚本中导入模型定义 ---
# 这样可以避免代码重复，并确保模型结构一致
try:
    from train_birwkv import BiRWKVConfig, BiRWKVModel
except ImportError:
    print("错误: 无法从 'train_birwkv.py' 导入模型。请确保该文件在同一目录下。")
    exit(1)

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def generate(
    model: BiRWKVModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_len: int = 64,
    num_steps: int = 12,
    device: torch.device = torch.device("cpu"),
):
    """
    使用迭代去噪方法从模型生成文本。

    Args:
        model: 训练好的BiRWKV模型。
        tokenizer: Tokenizer实例。
        prompt: 输入的文本提示。
        max_len: 要生成的最大token数量。
        num_steps: 去噪/生成的总步数。
        device: 运行模型的设备。
    """
    model.eval()
    mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
    
    # 编码输入提示
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # 初始化要生成的序列，全部用[MASK]填充
    # 我们将prompt和masked部分拼接在一起
    masked_ids = torch.full((1, max_len), mask_token_id, device=device)
    input_ids = torch.cat([prompt_ids, masked_ids], dim=1)
    
    # 生成过程的可视化
    log.info(f"起始序列: {tokenizer.decode(input_ids[0])}")
    
    with torch.no_grad():
        for step in range(num_steps):
            # 定义一个从1到0的去噪时间表（这里使用简单的线性调度）
            # t=1 (完全噪声), t=0 (无噪声)
            t = 1.0 - (step / num_steps)
            
            # 获取模型输出
            logits = model(input_ids)
            
            # 只关注被掩码部分的logits
            masked_positions = (input_ids == mask_token_id)
            
            # 如果没有需要填充的mask，则提前结束
            if not masked_positions.any():
                log.info("所有掩码已填充，提前结束生成。")
                break

            # 获取被掩码位置的预测概率和最可能的token
            probs = F.softmax(logits[masked_positions], dim=-1)
            max_probs, predictions = torch.max(probs, dim=-1)
            
            # 将所有[MASK]替换为预测的token
            input_ids[masked_positions] = predictions
            
            # --- 低置信度重掩码策略 ---
            # 1. 计算在当前时间步t，我们应该保留多少个token（不被掩码）
            # 我们让已知token的数量随步数增加
            num_known = int(max_len * (1.0 - t))
            
            # 2. 从新生成的token中，找出置信度最低的那些
            # 我们需要重新掩码的数量是 (总生成长度 - 应该知道的数量)
            num_to_remask = max_len - num_known
            
            if num_to_remask > 0:
                # 获取所有生成token的概率
                generated_probs = max_probs[-max_len:]
                
                # 找到置信度最低的k个token的索引
                remask_indices = torch.topk(generated_probs, k=num_to_remask, largest=False).indices
                
                # 获取这些token在整个input_ids中的实际位置
                # 这部分逻辑比较复杂，我们简化一下：直接在生成的部分进行操作
                generated_part = input_ids[0, -max_len:]
                generated_part[remask_indices] = mask_token_id
                input_ids[0, -max_len:] = generated_part

            log.info(f"第 {step+1}/{num_steps} 步: {tokenizer.decode(input_ids[0, prompt_ids.shape[1]:])}")

    final_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return final_text


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"使用设备: {device}")

    # --- 加载Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if '[MASK]' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['[MASK]'])
    
    # --- 加载模型 ---
    log.info(f"从检查点加载模型: {args.checkpoint_path}")
    config = BiRWKVConfig(vocab_size=len(tokenizer))
    model = BiRWKVModel(config)
    
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    except FileNotFoundError:
        log.error(f"错误: 检查点文件未找到 at {args.checkpoint_path}")
        return
    except Exception as e:
        log.error(f"加载模型权重时出错: {e}")
        return
        
    model.to(device)
    log.info("模型加载成功。")

    # --- 生成文本 ---
    final_output = generate(
        model,
        tokenizer,
        args.prompt,
        max_len=args.max_len,
        num_steps=args.num_steps,
        device=device,
    )
    
    log.info("\n" + "="*50)
    log.info("最终生成结果:")
    log.info(final_output)
    log.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试和生成 BiRWKV-LLaDA 模型")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="训练好的模型检查点路径 (.bin 文件)。"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains.",
        help="用于开始生成的文本提示。"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=80,
        help="要生成的文本的最大长度（不包括提示）。"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=16,
        help="迭代生成的步数。"
    )
    
    args = parser.parse_args()
    main(args)