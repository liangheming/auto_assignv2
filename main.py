# from processors.ddp_apex_processor import DDPApexProcessor
from processors.ddp_mix_processor import DDPMixProcessor

# nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port 50008 main.py >> temp.log 2>&1 &

if __name__ == '__main__':
    processor = DDPMixProcessor(cfg_path="config/auto_assign.yaml")
    processor.run()
