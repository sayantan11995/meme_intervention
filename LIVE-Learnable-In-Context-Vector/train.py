import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

from icv_src.icv_datamodule import VQAICVDataModule
from icv_src.icv_module import VQAICVModule
from utils import get_icv_cpk_path, init_interface

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="config", config_name="train.yaml")
def main(cfg: DictConfig):
    global logger
    pl.seed_everything(cfg.seed)
    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)

    model_name = cfg.lmm.name.split("/")[-1]
    save_path = get_icv_cpk_path(
        result_dir=cfg.result_dir,
        model_name=model_name,
        dataset_name=cfg.data_cfg.task.datasets.name,
        run_name=cfg.run_name,
    )

    save_path = Path(save_path)
    if (save_path / "icv_cpk.bin").exists():
        logger.info(f"{str(save_path / 'icv_cpk.bin')} exists! exit...")
        return
    wb_logger = WandbLogger(
        save_dir=cfg.result_dir,
        name=cfg.run_name,
        project="VQAInContextVector",
        log_model=False,
    )
    wb_logger.log_hyperparams(dict(cfg))
    trainer = pl.Trainer(
        logger=wb_logger,
        callbacks=[
            LearningRateMonitor(),
            RichModelSummary(max_depth=2),
            RichProgressBar(),
        ],
        **cfg.trainer,
        enable_checkpointing=False,
    )
    prompt_manager, interface, processor = init_interface(cfg)

    model = VQAICVModule(
        interface=interface, module_cfg=cfg.icv_module, lmm_cfg=cfg.lmm
    )
    data_module = VQAICVDataModule(
        data_cfg=cfg.data_cfg, prompt_manager=prompt_manager, prompt_processor=processor
    )

    trainer.fit(
        model,
        data_module,
    )
    trainer.save_checkpoint(
        filepath=os.path.join(
            save_path,
            "last.ckpt",
        ),
        weights_only=True,
    )
    postprocess(cfg, save_path)


@rank_zero_only
def postprocess(cfg, save_path):
    # TODO: Save layer map
    save_path = Path(save_path)
    if "deepspeed" in cfg.trainer.strategy:
        cpk_save_path = save_path / "last.ckpt"
        output_file = save_path / "lightning_module.bin"
        convert_zero_checkpoint_to_fp32_state_dict(cpk_save_path, output_file)

        checkpoint = torch.load(output_file)
        params_name = list(checkpoint["state_dict"].keys())
        for name in params_name:
            if "lmm" in name or "interface.model" in name:
                checkpoint["state_dict"].pop(name)
        checkpoint["state_dict"]["use_sigmoid"] = getattr(
            cfg.icv_module.icv_encoder, "use_sigmoid", None
        )
        checkpoint["state_dict"]["lmm_args"] = checkpoint["hyper_parameters"]["lmm_cfg"]
        torch.save(checkpoint["state_dict"], save_path / "icv_cpk.pth")
        os.remove(output_file)
        shutil.rmtree(
            cpk_save_path,
        )


if __name__ == "__main__":
    load_dotenv()
    main()
