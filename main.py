from models.build import build_model
from utils.parser import parse_args, load_config
from trainer import build_trainer
from utils.misc import mkdir, set_seeds, set_devices
from models.oracle.detector import DetectorOracleAD

def main():
    args = parse_args()
    cfg = load_config(args)

    # select cuda devices
    set_devices(cfg.VISIBLE_DEVICES)

    with open(mkdir(cfg.RESULT_DIR) / 'config.txt', 'w') as f:
        f.write(cfg.dump())

    # random seed 5회 평균 점수 계산으로 수정
    set_seeds(cfg.SEED)

    # build model
    model = build_model(cfg)

    # build trainer
    trainer = build_trainer(cfg, model)
    trainer.train()


if __name__ == '__main__':
    main()
