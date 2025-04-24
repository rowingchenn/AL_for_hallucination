#! /bin/bash

python /home/weichenzhang/hallucination/AL_for_hallucination/hallucination_traj_sample/WorkArena/generate_info.py \
    --output unachievable_easier_info.json \
    --truncate-map unachievable_easier_truncate_map.json

python /home/weichenzhang/hallucination/AL_for_hallucination/hallucination_traj_sample/WorkArena/generate_info.py \
    --output unachievable_info.json \
    --truncate-map unachievable_truncate_map.json

python /home/weichenzhang/hallucination/AL_for_hallucination/hallucination_traj_sample/WorkArena/generate_info.py \
    --output error_transition_info.json \
    --truncate-map error_transition_truncate_map.json