import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
import wandb

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    trainer = Trainer(config=cfg)
    with wandb.init(config=cfg,
                    project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    name=cfg.wandb.name,
                    group=cfg.wandb.group,
                    tags=cfg.wandb.tags,
                    notes=cfg.wandb.notes,
                    monitor_gym=cfg.wandb.monitor_gym):
        trainer.train(wandb=wandb)

if __name__ == "__main__":
    main()