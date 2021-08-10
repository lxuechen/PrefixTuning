nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/66234abb-620b-457c-a9b4-cfef19c55639'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/5b1dd74d-9859-4074-8559-6f727fbc9370'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/e2b94efb-090f-4555-969d-961a1b88f514'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/8f3edefc-4bca-4642-a92c-75eee1e5efeb'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/34656cb0-ac02-4e0b-9538-ba6473625b34'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/21b796ce-cf3d-44c1-b567-72d4fd768e19'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/9895b41d-59eb-4c96-bc90-6083d0b17676'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/dd86149c-35ce-418d-a9f9-ef99f3ebed18'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/97a5f385-59e4-4637-be10-9d20655a4a28'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/0db4f21a-8c59-4665-af69-2e82c1c9be9d'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/e11104ff-ca92-4c5f-9a4a-350677016bff'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/cd582064-8bee-452c-ae0e-861b3d5dcf48'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/30e47356-3c78-43cb-8632-58e553467ba7'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/8f82a9a0-76e1-49ee-994d-f741f3475005'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/73f1fab3-6e7e-450d-b944-a11f79ecb858'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0721/model_name_gpt2-medium_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000010_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/f539bdda-af34-4dd9-89ef-5257f5acef9c'
