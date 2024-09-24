lm_eval --model hf \
    --model_args pretrained=dfurman/Qwen2-72B-Orpo-v0.1,parallelize=True,attn_implementation="flash_attention_2",dtype="bfloat16",load_in_8bit=True,device_map="auto" \
    --apply_chat_template \
    --system_instruction "You are a helpful assistant." \
    --tasks leaderboard \
    --batch_size 1 \
    --log_samples \
    --output_path results \
    --hf_hub_log_args hub_results_org=dfurman,hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=Trues