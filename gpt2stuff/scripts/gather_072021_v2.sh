nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/65a7d7e4-9199-4ddd-94d1-fbff90259e90 --ref_path /nlp/scr/lxuechen/data/prefix-tuning/data/dart/json_clean_references_test.json'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_fulltune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/41c76454-6543-46fe-b5e4-ad376489f5c5 --ref_path /nlp/scr/lxuechen/data/prefix-tuning/data/dart/json_clean_references_test.json'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/f95ad314-1957-467c-9802-f30f1f902874 --ref_path /nlp/scr/lxuechen/data/prefix-tuning/data/dart/json_clean_references_test.json'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_scratchtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/a86e345d-0ee8-4457-93a1-ebfd34a9c683 --ref_path /nlp/scr/lxuechen/data/prefix-tuning/data/dart/json_clean_references_test.json'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/b3404069-2388-4dab-a974-692159fad498 --ref_path /nlp/scr/lxuechen/data/prefix-tuning/data/dart/json_clean_references_test.json'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_prefixtune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/259941ed-09d5-41ab-9262-3ca0ff1a6d59 --ref_path /nlp/scr/lxuechen/data/prefix-tuning/data/dart/json_clean_references_test.json'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/log_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/85c23e5d-0036-4f4b-ae44-8474e3fab783 --ref_path /nlp/scr/lxuechen/data/prefix-tuning/data/dart/json_clean_references_test.json'
nlprun -x=jagupard10,jagupard11,jagupard12,jagupard13,jagupard14,jagupard15,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27 -a lxuechen-prefix-tuning -o /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/log_ema_cpu.out -p low --memory 16g -g 0 'python -m gpt2stuff.eval.eval_generations --task eval_trajectory --gen_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_ema_model/eval --img_dir /nlp/scr/lxuechen/prefixtune/date_0720/model_name_gpt2_nonprivate_yes_tuning_mode_lineartune_learning_rate_0_00005000_train_batch_size_00000005_mid_dim_00000512_preseqlen_00000010_epochs_00000005_target_epsilon_-0000001/0/generations_ema_score --scratch_dir /nlp/scr/lxuechen/scratch/tmp/d8848b92-1fa7-4cb1-9b84-337b550d0305 --ref_path /nlp/scr/lxuechen/data/prefix-tuning/data/dart/json_clean_references_test.json'
