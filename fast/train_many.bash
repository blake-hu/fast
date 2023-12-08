#!/bin/bash

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model="$2"
            shift 2
            ;;
        --embedding)
            embedding_types=($2)
            shift 2
            ;;
        --tasks)
            tasks=($2)
            shift 2
            ;;
        --help)
            echo "Usage: $0 --model <model_name> --embedding <embedding_type1 embedding_type2 ...> --tasks <task1 task2 ...>"
            exit 1
            ;;
        *)
            echo "Invalid argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$model" ] || [ ${#embedding_types[@]} -eq 0 ] || [ ${#tasks[@]} -eq 0 ]; then
    echo "Missing or invalid arguments. Use --help for usage information."
    exit 1
fi

for embed in "${embedding_types[@]}"; do
    for task in "${tasks[@]}"; do

        echo "======= running: ${model} ${embed} ${task} ======="
        python3 run_task.py --model "$model" --embedding "$embed" --task "$task" --device "cuda:1"

    done
done
