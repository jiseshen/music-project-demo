# unidaw-demo
This private repo reflects my contribution to project UniDAW by 2025/12

## closed-source-probe
This used to be an attempt of distillation and scalable evaluation of proprietary music generation. suno was really hard to automate, since the repo suno-api had been out of date, not to mention it involved Turnstile and periodic token update. My patch used to work for a while but automation was soon made impossible again by Suno.

## benchmark
One of the major contribution. This contains all the code for 8 baselines and 14 metrics across 5 different tasks. It supports incremental testing and is compatible to multiple data structures. visqol was hard to adapt, since it relied on bazel and was no longer maintained. MuMuLLaMA's released ckpt has missing parameters. 

## ddsp
Exploration of potential timber transfer teacher models. But because of limited timbers and bad out-of-domain performance that was not used. Some effort has been made to relocate the checkpoint files.

## demucs_batch-gpu
Official Demucs repo didn't support batch processing. This is an adapted one which can scale up using batched input and distributed run. This managed to carry FMA (8k hours) source seperation on 3 nodes in 12 hours, which would otherwise take 3 days.

## clamp_encodec_cluster
One of the major contribution. This adapted Meta's Dora framework and patched encodec training logic to support contrastive learning with Clamp model, in an effort of achieving UniTok. Dora framework is hyper-parameter-oriented, migration and adaption took a while.

## train-script-modify
After we have access to Marlowe's 8-H100 clusters, we found that our code does not work well on it - it took around 3 days to train one run of our full episode. I first worked on figuring out CPU-GPU overheads and amortized by finding the near-optimal configuration of DataLoader and Distributed pipeline. I fixed test-time memory leakage by forcing no-cache model inference and adjusted aggregation of cross-GPU loss data. Now the training can finish in 1 day.
