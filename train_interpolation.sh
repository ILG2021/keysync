filelist=$1
workers=${2:-6}
batch_size=${3:-1}
device=${4:-1}
python main.py --base configs/example_training/interpolation/interpolation_dub.yaml --wandb True lightning.trainer.num_nodes=1 \
    lightning.strategy=deepspeed_stage_1 lightning.trainer.precision=32 model.base_learning_rate=3.e-5 \
    data.params.train.datapipeline.filelist=$filelist \
    data.params.train.datapipeline.video_folder=video_crop  \
    data.params.train.datapipeline.audio_folder=audio \
    data.params.train.datapipeline.audio_emb_folder=audio_emb \
    data.params.train.datapipeline.latent_folder=video_crop_emb \
    data.params.train.datapipeline.landmarks_folder=landmarks_crop \
    data.params.train.loader.num_workers=$workers \
    data.params.train.datapipeline.audio_in_video=False \
    data.params.train.datapipeline.load_all_possible_indexes=False \
    lightning.trainer.devices=$device lightning.trainer.accumulate_grad_batches=1 data.params.train.datapipeline.select_randomly=False \
    model.params.network_config.params.audio_cond_method=both_keyframes data.params.train.datapipeline.what_mask=box \
    'model.params.to_freeze=["time_"]' 'model.params.to_unfreeze=["time_embed"]' data.params.train.loader.batch_size=$batch_size  data.params.train.datapipeline.audio_emb_type=hubert \