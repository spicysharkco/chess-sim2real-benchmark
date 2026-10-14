# C4 end-to-end latency

Fixed C1 dev selection: 250 scenes / 500 parity-paired images, 192 unique position groups (58 repeated group-scenes).
Hardware: NVIDIA GeForce RTX 3090; fixed power limit 270.00 W; driver 595.71.05.
Batch size 1, native FP16 autocast, 30 excluded warm-up images. File decode is excluded; native resize/detector/router/warp/TTA/argmax/host transfer/FEN construction are included as applicable.
Private variants retain different native members, routing and TTA, so these timings do not compare equal FLOPs.

| Public model alias | Median ms/image | p90 ms/image |
|---|---:|---:|
| cq_release | 64.414 | 65.724 |
| private_full | 195.516 | 244.216 |
| private_raw | 79.694 | 81.072 |
| private_warp_high | 195.231 | 197.550 |
| private_warp_low | 168.984 | 170.853 |
| lora_dark_n500_s303 | 71.876 | 72.642 |
| lora_both_n500_s303 | 72.403 | 73.316 |
| lora_dark_n2000_s303 | 72.292 | 73.077 |
| lora_both_n2000_s303 | 71.501 | 72.496 |
| lora_dark_n2000_s304 | 71.864 | 72.948 |
| lora_both_n2000_s304 | 72.795 | 73.675 |
| lora_dark_n2000_s305 | 72.500 | 73.477 |
| lora_both_n2000_s305 | 73.121 | 74.025 |
| lora_dark_n8000_s303 | 72.403 | 73.257 |
| lora_both_n8000_s303 | 72.655 | 73.568 |

These measurements describe this fixed hardware/software/input pipeline only; no accuracy, superiority, equal-compute, or general latency conclusion follows automatically.
