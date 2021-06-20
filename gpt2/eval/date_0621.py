import os
import re

from lxuechen_utils import utils

date = "0621"
model_name_or_path = 'distilgpt2'
nonprivate = "no"
tuning_mode = "prefixtune"
per_example_max_grad_norm_str = utils.float2str(0.1)
noise_multiplier_str = utils.float2str(0.7)
mid_dim_str = utils.int2str(512)
preseqlen_str = utils.int2str(10)
seed = 0

out_dir = f"/nlp/scr/lxuechen/plots/{date}"

for train_batch_size in (100, 200, 300, 400, 500):
    for lr in (3e-3, 1e-3, 3e-4, 1e-4):
        learning_rate_str = utils.float2str(lr)
        train_batch_size_str = utils.int2str(train_batch_size)

        train_dir = (
            f"/nlp/scr/lxuechen/prefixtune/date_{date}"
            f"/model_name_{model_name_or_path}_nonprivate_{nonprivate}_tuning_mode_"
            f"{tuning_mode}_per_example_max_grad_norm_{per_example_max_grad_norm_str}_noise_multiplier_"
            f"{noise_multiplier_str}_learning_rate_{learning_rate_str}_train_batch_size_{train_batch_size_str}_mid_dim_"
            f"{mid_dim_str}_preseqlen_{preseqlen_str}"
            f"/{seed}/generations/eval"
        )

        if not os.path.exists(train_dir):
            continue

        global_steps = []
        for file_path in utils.list_file_paths(train_dir):
            if re.match(".+global_step_.+.txt", file_path):
                global_steps.append(
                    int(file_path[-12:-4]),
                )
        global_steps.sort()
        global_steps = [str(gs) for gs in global_steps]
        global_steps = ", ".join(global_steps)

        img_dir = os.path.join(out_dir, f"{train_batch_size_str}-{learning_rate_str}")
        os.makedirs(img_dir, exist_ok=True)
        os.system(
            f"python -m gpt2.eval.eval_generations --task eval_trajectory --gen_dir {train_dir} --global_steps "
            f"{global_steps} --img_dir {img_dir}")

# python -m gpt2.eval.date_0621
