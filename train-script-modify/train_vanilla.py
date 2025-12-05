# [Yang Xu] modify from my personal project with the framework of deepspeed & accelerator training

import os
import sys
import random

from tqdm import tqdm

from accelerator_models import FlatAudioTokenModel
import json
import transformers

import torch

from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
# from torch.utils.data.distributed import DistributedSampler  # not used

torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# from datasets import Dataset  # not used here
from datetime import datetime

from utils import (
    get_latest_checkpoint,
    randomize_with_seed,
    set_momentum,
    parse_args,
    get_token_datasets,
)


def main():
    # formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    # 1. initialize the training
    if args.seed == 0:
        args.seed = int(datetime.now().timestamp())
    randomize_with_seed(args.seed)

    mixed_precision = "fp16" if args.fp16 else "bf16"
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    set_seed(args.seed)

    """
    prepare model
    """
    _step = 0
    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
        print(
            f"[INFO] Checkpoint path specified, and the step will be regarded as 0. Loading checkpoint from {checkpoint_path}..."
        )
    else:
        checkpoint_path = None
        if not args.train_from_scratch:
            with accelerator.main_process_first():  # avoid multiple processes to generate the same checkpoint
                checkpoint_path, _step = get_latest_checkpoint(
                    args.exp_name, args.test_on_epoch, args.test_on_step
                )  # TODO: add the exp name
    # _epoch += 1  # correct the right epoch number
    # Model construction does not need autocast; Accelerate will handle mixed precision during forward
    osp = FlatAudioTokenModel(
        args.model_name,
        device=accelerator.device,
        hf_token=hf_token,
        local_checkpoint_path=(
            checkpoint_path
            if (checkpoint_path is not None and checkpoint_path.endswith("bin"))
            else None
        ),
        # load chkpt from local path iff the checkpoint path is a local file, if it's a dir, we'll load that from accelerate.load_state
        audio_token_vocab_size=args.audio_token_vocab_size,
        expand_features=1 if args.single_channel else args.first_num_features,
        dropout=args.dropout,  # set dropout
        use_flash_attention=not args.fp16,
        audio_encodec_path=args.audio_tokenizer,
        need_translate=args.train_with_translate,
        
    )

    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     target_modules=["q_proj", "v_proj"],  # Qwen typically uses these
    #     lora_dropout=args.dropout,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )

    # osp.base_model = get_peft_model(osp.base_model, lora_config)
    tokenizer = osp.base_model_tokenizer
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, osp.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )

    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optim,
    #     T_max=args.num_epochs * len(osp.train_dataloader()),
    #     eta_min=0.0,
    # )

    # resize model vocab
    # osp.resize_token_embeddings(len(tokenizer) + args.audio_token_vocab_size + 1)

    # if args.checkpoint_path and not args.test_only:
    #     optim_chkpt = (
    #         f"zero_pp_rank_{accelerator.process_index}_mp_rank_00_optim_states.pt"
    #     )
    #     optim_dir = args.checkpoint_path.rsplit("/", maxsplit=1)[0]
    #     optim_state_dict = torch.load(os.path.join(optim_dir, "pytorch_model", optim_chkpt))
    #     optim.load_state_dict(optim_state_dict)

    """
    prepare data
    """
    if args.dataset_path is None:
        accelerator.print(
            "Loading datasets under folder dataset, searching for " + args.dataset_name
        )
    else:
        accelerator.print(
            "[INFO] Dataset path is specified. Force loading dataset from: "
            + args.dataset_path
        )
        args.dataset_name = None
    truncate_lens = None
    if args.truncate_lens is not None:
        truncate_lens = args.truncate_lens.split(",")
        truncate_lens = list(map(lambda x: int(float(x) * 50), truncate_lens))
    with accelerator.main_process_first():  # avoid multiple processes to generate the same dataset
        dataloaders = get_token_datasets(
            args=args,
            dataset_names=args.dataset_name,
            dataset_path=args.dataset_path,
            bos_token=tokenizer.bos_token_id,
            pad_token=tokenizer.pad_token_id,
            truncate_frame_num=truncate_frame_num,
            truncate_lens=truncate_lens,
            num_processes=accelerator.num_processes,
            process_index=accelerator.process_index,
            text_tokenizer=tokenizer if "Llama-3.2-1B" not in args.model_name else None,
        )
        sep_dataloaders = get_token_datasets(
            args=args,
            dataset_names=args.dataset_name,
            dataset_path=args.dataset_path,
            bos_token=tokenizer.bos_token_id,
            pad_token=tokenizer.pad_token_id,
            truncate_frame_num=truncate_frame_num,
            truncate_lens=truncate_lens,
            num_processes=accelerator.num_processes,
            process_index=accelerator.process_index,
            train_sep=True,
            text_tokenizer=tokenizer if "Llama-3.2-1B" not in args.model_name else None
        )
    train_loader, valid_loader, sep_train_loader, sep_valid_loader = (
        dataloaders["train"],
        dataloaders["valid"],
        sep_dataloaders["train"],
        sep_dataloaders["valid"],
    )
    assert len(train_loader) >= len(sep_train_loader)  # common sense
    for split, loader in zip(
        ["train", "valid", "train_sep", "valid_sep"],
        [train_loader, valid_loader, sep_train_loader, sep_valid_loader],
    ):
        accelerator.print(f"[INFO] {split} dataset has {len(loader.dataset)} samples")

    if args.use_wandb:
        # wandb.login()
        assert os.environ.get(
            "WANDB_API_KEY"
        ), "Please set the WANDB_API_KEY environment variable in advance, we do not want to manually log in"
        accelerator.init_trackers(
            # Set the project where this run will be logged
            project_name="MMIO",
            # Track hyperparameters and run metadata
            config={
                "dataset": (
                    args.dataset_name
                    if args.dataset_path is None
                    else args.dataset_path
                ),
                "batch_size": args.batch_size,
                "seconds": args.data_truncate_seconds,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "len_train_loader": len(sep_train_loader) * 2,
                "seed": args.seed,  # Consider the case where resume from another checkpoint, avoid shuffling with the same seed
            },
            init_kwargs={
                "wandb": {
                    "entity": "mmio_team",
                    "name": args.exp_name
                    + f"_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
                }
            },
        )

    sched = None
    if args.use_lr_scheduler:
        if args.scheduler_total_steps < 0:
            args.scheduler_total_steps = args.num_epochs * len(sep_train_loader) * 2
            scheduler_total_steps = (
                args.scheduler_total_steps
            )  # * accelerator.num_processes
        else:
            scheduler_total_steps = (
                args.scheduler_total_steps * accelerator.num_processes
            )
        args.warm_up_steps *= accelerator.num_processes
        accelerator.print("Scheduler steps:", args.scheduler_total_steps)
        # [Deprecated] accelerator.print("Note this will be multiplied by number of processes.")
        # set min lr
        sched = transformers.get_cosine_schedule_with_warmup(
            optimizer=optim,
            num_warmup_steps=args.warm_up_steps,
            num_training_steps=scheduler_total_steps,
            # min_lr=args.min_learning_rate,  # set the minimum learning rate
        )

    """
    prepare accelerator
    """
    (
        osp,
        optim,
        train_loader,
        valid_loader,
        sep_train_loader,
        sep_valid_loader,
        sched,
    ) = accelerator.prepare(
        osp,
        optim,
        train_loader,
        valid_loader,
        sep_train_loader,
        sep_valid_loader,
        sched,
    )

    if checkpoint_path is not None and not checkpoint_path.endswith(
        ".bin"
    ):  # it's a dir
        # load the state from the checkpoint dir
        # accelerator.print(f"Loading state from checkpoint dir: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)

    # 3. start to train

    # prefix steps
    # if args.from_step > 0:
    #     prefix_steps = args.from_step
    # else:
    #     prefix_steps = args.from_epoch * len(train_loader)

    accelerator.print(f"*****************************")
    set_momentum(optim)
    osp.train()

    # accelerator.print(optim, optim.param_groups)
    os.makedirs(f"./outputs/{args.exp_name}", exist_ok=True)
    # log_mode = "a" if checkpoint_path else "w"
    # with open(
    #     f"./outputs/{args.exp_name}/train_log.txt", log_mode, encoding="utf-8"
    # ) as log_file:
    with accelerator.autocast():
        for epoch in range(args.num_epochs):
            pbar = tqdm(
                range(len(sep_train_loader) * 2),
                disable=not accelerator.is_main_process,
            )
            # accelerator.print(f"*************[EPOCH {epoch + _epoch}]**************")
            for batch_index, (batch_data_, sep_batch_data_) in enumerate(
                zip(train_loader, sep_train_loader)
            ):
                current_step = epoch * len(sep_train_loader) * 2 + batch_index * 2 + _step - 1
                if args.use_lr_scheduler and current_step + 1 >= args.scheduler_total_steps:
                    accelerator.print(
                        f"Reached the maximum training steps {args.scheduler_total_steps}, stopping training."
                    )
                    break
                # train_loss = 0.0
                # consider the data batch, combining text tokens and audio tokens
                # seq = text_tokens * 0
                # mask = text_tokens_mask
                # if seq.shape[-1] > 300:
                #     continue
                for batch_data in (batch_data_, sep_batch_data_):
                    current_step += 1
                    with accelerator.accumulate(osp):
                        # Note osp is just a common llama model
                        if batch_data.get("other_token_ids") is not None:
                            task = random.choices(args.tasks, weights=args.tasks_ratio)[0]
                        else:
                            task = random.choices(
                                args.no_sep_tasks, weights=args.no_sep_tasks_ratio
                            )[0]
                        # with accelerator.autocast():
                        outputs = osp(batch_data, task=task)

                        train_loss = outputs["loss"]
                        pbar.set_postfix(
                            {
                                f"train_loss": train_loss.item(),
                            }
                        )

                        accelerator.backward(train_loss)

                        # [DEBUG] Check whether if the loss is nan
                        assert torch.isnan(train_loss).sum() == 0, print(train_loss)

                        accelerator.clip_grad_norm_(
                            list(osp.parameters()),
                            args.max_grad_clip_norm,
                            norm_type=2,
                        )
                        if args.use_wandb:
                            train_log_dict = {f"{task}_train_loss": train_loss.item()}
                            # TODO: more loss info
                            if sched:
                                train_log_dict.update({"learning_rate": sched.get_lr()[0]})
                            accelerator.log(
                                train_log_dict,
                                step=current_step,
                            )
                        optim.step()
                        if sched:
                            sched.step()
                        optim.zero_grad()
                        accelerator.wait_for_everyone()
                    pbar.update(1)
                    # torch.cuda.empty_cache()

                    # eval after certain step (TODO: add validation for fixed number of steps)
                    if (current_step + 1) % args.eval_by_step == 0:
                        eval_step = current_step + 1
                        osp.eval()
                        eval_log_dict = {}
                        for task in args.tasks:
                            # Accumulate on CPU to avoid GPU scalar tensor growth
                            loss_sum = 0.0
                            num_batches = 0
                            if task in ["source_separation", "instrument_infill"]:
                                val_loader = sep_valid_loader
                            else:
                                val_loader = valid_loader
                            with torch.inference_mode():
                                for val_batch_data in tqdm(val_loader, disable=not accelerator.is_main_process):
                                    # with accelerator.autocast():
                                    outputs = osp(val_batch_data, task=task)
                                    loss_sum += outputs["loss"].item()  # move to CPU immediately
                                    num_batches += 1

                            # Average locally, then average across processes
                            avg_loss_local = loss_sum / max(1, num_batches)
                            avg_loss_tensor = torch.tensor(avg_loss_local, device=accelerator.device)
                            avg_loss = accelerator.gather(avg_loss_tensor).mean()
                            accelerator.print(
                                f"Eval step {eval_step}, {task} valid loss: {avg_loss.item()}"
                            )
                            eval_log_dict.update(
                                {
                                    f"{task}_valid_loss": avg_loss.item(),
                                }
                            )
                            accelerator.wait_for_everyone()
                        if args.use_wandb:
                            accelerator.log(
                                eval_log_dict,
                                step=eval_step,
                            )
                        osp.train()
                        # torch.cuda.empty_cache()
                        accelerator.wait_for_everyone()

                    if (current_step + 1) % args.save_by_step == 0:
                        accelerator.save_state(
                            output_dir=f"./outputs/{args.exp_name}/step_{current_step+1}"
                        )

        accelerator.end_training()


if __name__ == "__main__":
    # parse args and set global vars
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "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")
    setattr(args, "hf_token", hf_token)
    # torch_dtype is not needed when Accelerate manages mixed precision
    truncate_frame_num = int(args.data_truncate_seconds * 50)  # 50 frames per second
    main()
