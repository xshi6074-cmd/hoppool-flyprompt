import argparse

from datasets import DATASETS
from methods import METHODS


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # ========== Experiment configuration ==========
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--note", type=str, default="", help="Short description of the exp")
    parser.add_argument("--log_path", type=str, default="results", help="The path logs are saved.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=(
            "Physical GPU ID to expose to this process. When specified, the "
            "selected device is available inside Python as cuda:0."
        ),
    )

    # ============ Model configuration =============
    parser.add_argument("--method", type=str, default="l2p", help="Method name", choices=METHODS.keys())
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224", help="Backbone name")
    parser.add_argument(
        "--backbone_path",
        type=str,
        default=None,
        help=(
            "Path to a local pretrained backbone checkpoint. This initializes "
            "the backbone only; it does not resume optimizer or training state."
        ),
    )

    # =========== Dataset configuration ============
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name", choices=DATASETS.keys())
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset root directory (see README for expected layouts per dataset).")
    parser.add_argument("--n_tasks", type=int, default=5, help="The number of tasks")
    parser.add_argument("--step_num", type=int, default=-1,
                        help="Number of internal steps for task-free prompt methods; if <=0, defaults to n_tasks.")

    parser.add_argument("--n", type=int, default=50, help="The percentage of disjoint split. Disjoint=100, Blurry=0")
    parser.add_argument("--m", type=int, default=10, help="The percentage of blurry samples in blurry split. Uniform split=100, Disjoint=0")
    parser.add_argument("--rnd_NM", action='store_true', default=False, help="if True, N and M are randomly mixed over tasks.")

    # =========== Training configuration ===========
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision.")
    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epoch.")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")

    parser.add_argument("--transforms", nargs="*", default=['cutmix', 'autoaug'], help="Additional train transforms [cutmix, cutout, autoaug]")
    parser.add_argument("--no_batchmask", action="store_true", default=False, help="Disable batch mask, use seen mask")

    # ========== Evaluation configuration ==========
    parser.add_argument("--topk", type=int, default=1, help="set k when we want to set topk accuracy")
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")

    # ============= ViT configurations =============
    parser.add_argument('--profile', action='store_true', default=False, help='enable profiling for ViT_Prompt')

    # ============= MISA configurations ============
    parser.add_argument('--load_pt', action='store_true', default=False, help='load pretrained prompts (MISA)')

    # ============= MePo configurations ============
    parser.add_argument('--mepo_backbone_path', type=str, default=None,
                        help='Path to pretrained backbone checkpoint for MEPO backbone override.')
    parser.add_argument('--cov_path', type=str, default=None,
                        help='Path to covariance matrix .npy for MEPO CLS calibration.')
    parser.add_argument('--cov_coef', type=float, default=0.7,
                        help='Interpolation coeff between original and MEPO-calibrated CLS (0-1).')

    # ======== HiDe / NoRGa configurations =========
    parser.add_argument("--lam_orth", type=float, default=1, help="Orthogonal loss weight for HiDe/NoRGa.")
    parser.add_argument("--ca_num_per_class", type=int, default=200, help="Number of CA samples per class for HiDe/NoRGa.")
    parser.add_argument("--ca_steps", type=int, default=200, help="Number of CA optimization steps for HiDe/NoRGa.")

    # ========== SD-LoRA configurations ==========
    parser.add_argument("--sdlora_rank", type=int, default=10, help="LoRA rank for SD-LoRA (default from original SD-LoRA).")
    parser.add_argument("--sdlora_alpha", type=float, default=0.8, help="Scaling factor alpha for SD-LoRA (default from original SD-LoRA).")
    parser.add_argument("--sdlora_layers", type=str, default="all", help="Which ViT blocks to apply LoRA to (e.g., 'all', 'last4').")
    parser.add_argument("--sdlora_ortho_weight", type=float, default=0.0, help="Orthogonal loss weight for SD-LoRA (0 means disabled).")

    # ========== FlyPrompt configurations ==========
    parser.add_argument("--len_prompt", type=int, default=20, help="The length of the prompt for each expert")
    parser.add_argument("--pos_prompt", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="The position of the prompt")
    parser.add_argument("--rp_dim", type=int, default=10000, help="The dimension of the random projection head")
    parser.add_argument("--rp_ridge", type=float, default=1e4, help="The ridge parameter for the random projection head")
    parser.add_argument("--use_ema", action="store_true", default=False,
                        help=(
                            "Ensemble EMA classifier heads with the online FC: one bank per "
                            "expert for flyprompt, one global bank for baseline, shared_prompt, "
                            "hfpool and gate. Without it only the online FC is used."
                        ))
    parser.add_argument("--ema_ratio", type=float, nargs="+", default=[0.9, 0.99], help="The EMA ratio for the expert FCs")
    parser.add_argument("--ensemble_method", type=str, default="softmax_max_prob", choices=["mean", "max_prob", "min_entropy", "softmax_mean", "softmax_max_prob", "softmax_min_entropy"],
                        help="Ensemble method for combining expert outputs: mean (average), max (maximum), min_entropy (minimum entropy), and softmax variants of these.")
   
    # ========== RPFC gating configurations ==========
    parser.add_argument("--use_rp_gate", action="store_true", default=False,
                        help="Use FlyPrompt-style RPFC head for task gating in compatible methods (e.g., SPrompt, HiDe/NoRGa, DualPrompt, MVP).")

    # ========== EMA head bank configurations ==========
    parser.add_argument("--use_ema_head", action="store_true", default=False,
                        help="Use EMA-based classifier head bank and ensemble in compatible methods (e.g., SPrompt, HiDe/NoRGa, DualPrompt, MVP).")

    
    parser.add_argument("--analysis_expert_similarity", action="store_true", default=False,
                        help="If set, run expert feature similarity / CKA (including residual vs common) analysis after training.")

    # ========== MLP generator configurations ==========
    parser.add_argument("--prompt_blocks", type=int, nargs="+", default=[0],
                        help="Strictly increasing ViT block indices that receive generated prompts.")
    parser.add_argument("--generator_layers", type=int, choices=[1, 2, 3], default=1,
                        help="Number of Linear layers in the clean-CLS prompt generator.")
    parser.add_argument("--generator_hidden_dim", type=int, default=None,
                        help="Hidden width for 2/3-layer generators; defaults to the backbone embedding width.")
    parser.add_argument("--distill_weight", type=float, default=1.0,
                        help=(
                            "Weight of old-class prompt functional distillation. "
                            "0 disables the teacher branch entirely (no anchor "
                            "sampling, no teacher forward): the no-distillation "
                            "control in fixed mode. In uncertainty mode this is "
                            "the initial value."
                        ))
    parser.add_argument("--distill_weight_mode",
                        choices=["fixed", "uncertainty", "gradnorm"],
                        default="fixed",
                        help=(
                            "How the distillation coefficient is chosen. fixed "
                            "uses --distill_weight; uncertainty keeps the legacy "
                            "exp(-s)*D+s objective; gradnorm uses a detached "
                            "one-sided gradient-ratio controller."
                        ))
    parser.add_argument("--learnable_distill_weight", action="store_true",
                        help=(
                            "Legacy alias for --distill_weight_mode uncertainty. "
                            "It conflicts with --distill_weight_mode gradnorm."
                        ))
    parser.add_argument("--distill_delay_samples", type=int, default=100,
                        help=(
                            "After every hard-teacher snapshot, skip the entire "
                            "distillation branch until at least this many global "
                            "stream samples have completed. Whole batches are not "
                            "split."
                        ))
    parser.add_argument("--distill_grad_ratio", type=float, default=0.25,
                        help=(
                            "GradNorm-lite target ||w grad D|| / ||grad CE|| on "
                            "the generator reference layer."
                        ))
    parser.add_argument("--distill_grad_ema", type=float, default=0.9,
                        help="EMA coefficient applied to log GradNorm weights.")
    parser.add_argument("--distill_weight_min", type=float, default=1e-4,
                        help="Lower bound for the GradNorm-lite target weight.")
    parser.add_argument("--distill_weight_max", type=float, default=1e4,
                        help="Upper bound for the GradNorm-lite target weight.")
    parser.add_argument("--distill_metric", choices=["mse", "cosine"], default="mse",
                        help="Distance between student and hard-teacher generated prompts.")
    parser.add_argument("--statistic_type", choices=["diagonal", "low_rank"], default="diagonal",
                        help="Class-conditional clean-CLS replay distribution.")
    parser.add_argument("--covariance_rank", type=int, default=16,
                        help="Rank of the correlated M2 sketch when statistic_type=low_rank.")
    parser.add_argument("--sketch_shrinkage", choices=["none", "fd"], default="none",
                        help=(
                            "How the low-rank sketch is compressed after each "
                            "merge. 'none' keeps the plain top-r truncation, "
                            "which is per-step optimal but lets the sketch "
                            "freeze on whichever directions were dominant "
                            "first. 'fd' subtracts the first discarded "
                            "eigenvalue (Frequent Directions), which unfreezes "
                            "new directions but strongly deflates the sketch "
                            "when the spectrum is flat relative to "
                            "--covariance_rank. Measure the spectrum before "
                            "choosing 'fd'."
                        ))
    parser.add_argument("--replay_class_budget", type=int, default=8,
                        help="Total old classes replayed per optimizer step across all active teachers.")
    parser.add_argument("--anchors_per_class", type=int, default=1,
                        help=(
                            "Anchors drawn per selected replay class. With 1 the "
                            "distillation loss is a rank-1 Monte-Carlo estimate of "
                            "the anchor second moment, so no covariance model can "
                            "be distinguished from a diagonal one. Use >= 8 when "
                            "ablating --statistic_type."
                        ))
    parser.add_argument("--replay_eligibility", choices=["teacher", "previous"],
                        default="previous",
                        help=(
                            "Which snapshot decides the replay-eligible classes. "
                            "'previous' (default) uses the snapshot one internal "
                            "step older than the teacher, so classes introduced "
                            "while the teacher itself was still learning them are "
                            "not replayed. 'teacher' uses the teacher's own "
                            "snapshot, which is the older behaviour and lets a "
                            "class under active training into the replay pool."
                        ))
    parser.add_argument("--distill_mode", choices=["sample", "closed_form"],
                        default="sample",
                        help=(
                            "How the teacher-student prompt discrepancy is "
                            "measured. 'sample' draws anchors from the stored "
                            "class distribution. 'closed_form' evaluates the "
                            "exact population MSE of the affine student-teacher "
                            "difference over that distribution, with no anchor "
                            "sampling and no estimator noise; it requires "
                            "--generator_layers 1 and --distill_metric mse, and "
                            "forces --statistics_space normalized."
                        ))
    parser.add_argument("--statistics_space", choices=["raw", "normalized"],
                        default="raw",
                        help=(
                            "Space in which clean-CLS class statistics are "
                            "collected. 'normalized' applies the generator's "
                            "parameter-free LayerNorm normalisation first, which "
                            "removes the shared mean direction that otherwise "
                            "dominates the second moment. Required by "
                            "--distill_mode closed_form."
                        ))
    parser.add_argument("--min_replay_samples", type=int, default=2,
                        help="Minimum class count at snapshot time before that class is replay-eligible.")
    parser.add_argument("--variance_prior_strength", type=float, default=16.0,
                        help="Count-based shrinkage strength toward pooled within-class diagonal variance.")
    parser.add_argument("--variance_floor", type=float, default=1e-5,
                        help="Minimum replay variance per clean-CLS dimension.")
    parser.add_argument("--variance_ceiling", type=float, default=10.0,
                        help="Maximum replay variance per clean-CLS dimension.")
    parser.add_argument("--teacher_lags", type=int, nargs="+", default=[1],
                        help="Hard-teacher snapshot lags measured in internal steps, e.g. 1 5.")

    # ========== HFPool configurations ==========
    parser.add_argument("--num_pooling_heads", type=int, default=1,
                        help="Attention heads per Hopfield pooling layer.")
    parser.add_argument("--num_pooling_blocks", type=int, default=5,
                        help="Number of consecutive ViT blocks that receive Hopfield prompts.")
    parser.add_argument("--prompt_length", type=int, default=10,
                        help="Prompt tokens pooled per Hopfield layer.")
    parser.add_argument("--deep_prompts", type=int, default=0,
                        help="1 places the pooling blocks at the deepest layers instead of the "
                             "shallowest; ignored when --hopfield_start_layer is set.")
    parser.add_argument("--hopfield_start_layer", type=int, default=None,
                        help="Zero-based first ViT block that receives a Hopfield prompt.")
    parser.add_argument("--hopfield_input_mode", choices=["local", "full_vit"], default="local",
                        help=(
                            "local pools from the tokens entering each prompted block; full_vit "
                            "(ablation) pools every layer from one detached clean full-ViT token "
                            "sequence and requires --online_iter 3."
                        ))
    parser.add_argument("--hopfield_trainable", nargs="+", choices=["query", "q", "k", "v", "o"],
                        default=["q"],
                        help=(
                            "Hopfield parts that receive gradients: query (pooling weights), "
                            "q/k/v (packed input projection slices), o (output projection). "
                            "V is a static identity unless v is listed, in which case an "
                            "identity-initialised projection is trained. Normalisation stays frozen."
                        ))
    parser.add_argument("--hopfield_grad_clip", action="store_true",
                        help="Clip trainable Hopfield gradients after AMP unscaling (stabilises joint training).")
    parser.add_argument("--hopfield_grad_clip_norm", type=float, default=1.0,
                        help="Maximum Hopfield gradient norm used with --hopfield_grad_clip.")

    # ========== Gate configurations ==========
    parser.add_argument("--gate_blocks", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="ViT blocks whose attention heads get a learned 1 + tanh(alpha) gain.")

    args = parser.parse_args()
    return args
