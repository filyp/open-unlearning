# %%
import os

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from data import get_collators, get_data
from evals import get_evaluators
from model import get_model

# %%
# Initialize hydra config
config_path = os.path.abspath("../configs")

with initialize_config_dir(version_base=None, config_dir=config_path):
    cfg = compose(
        config_name="unlearn.yaml",
        overrides=[
            "experiment=unlearn/wmdp_deduped/default",
            "trainer=CIR",
            "task_name=NOGRAD_NORETAIN_0.01_MAHAL_NODOWNPROJ",
        ],
    )

# print(OmegaConf.to_yaml(cfg))

model_cfg = cfg.model
mode = cfg.get("mode", "train")
template_args = model_cfg.template_args

model, tokenizer = get_model(model_cfg)
data = get_data(cfg.data, mode=mode, tokenizer=tokenizer, template_args=template_args)
collator = get_collators(cfg.collator, tokenizer=tokenizer)

# Get Trainer
trainer_cfg = cfg.trainer
assert trainer_cfg is not None, ValueError("Please set trainer")

# %%
# Load model

# model, tokenizer = get_model(model_cfg)
# model = model.to("cuda")

# Get Evaluators
evaluators = None
eval_cfgs = cfg.get("eval", None)
evaluators = get_evaluators(
    eval_cfgs=eval_cfgs,
    template_args=template_args,
    model=model,
    tokenizer=tokenizer,
    data=data,
)

# %%
from trainer.unlearn.cir.cir_trainer import *
from trainer import load_trainer

trainer, trainer_args = load_trainer(
    trainer_cfg=trainer_cfg,
    model=model,
    train_dataset=data.get("train", None),
    eval_dataset=data.get("eval", None),
    tokenizer=tokenizer,
    data_collator=collator,
    evaluators=evaluators,
    template_args=template_args,
)

# %%

trainer.train()
# %%
data["wikitext"][0]["input_ids"].device