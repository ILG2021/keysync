import gradio as gr
import os
import subprocess
import yaml
import time
import threading
import signal
import sys
from omegaconf import OmegaConf

# Roots
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
OUTPUT_DIR = os.path.join(ROOT_DIR, "gradio_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Environment with PYTHONPATH
ENV = os.environ.copy()
ENV["PYTHONPATH"] = ROOT_DIR + os.pathsep + ENV.get("PYTHONPATH", "")

# Training process tracking
training_process = None
training_logs_list = []

def log_reader(pipe):
    global training_logs_list
    try:
        for line in iter(pipe.readline, ''):
            if not line: break
            training_logs_list.append(line)
            if len(training_logs_list) > 200:
                training_logs_list.pop(0)
    except Exception:
        pass
    finally:
        pipe.close()

def get_checkpoints():
    ckpt_dir = os.path.join(ROOT_DIR, "pretrained_models", "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    return [f for f in os.listdir(ckpt_dir) ]

def preprocess_video(video_path):
    """Generate landmarks and latents if missing."""
    # Landmarks
    landmark_path = video_path.replace(".mp4", ".npy")
    if not os.path.exists(landmark_path):
        print(f"Generating landmarks for {video_path}...")
        cmd = [sys.executable, "scripts/util/gen_landmarks.py", os.path.dirname(video_path) if os.path.dirname(video_path) else ".", "--output_dir", os.path.dirname(video_path) if os.path.dirname(video_path) else "."]
        subprocess.run(cmd, env=ENV, cwd=ROOT_DIR)
    
    # Latents
    latent_path = video_path.replace(".mp4", "_video_512_latent.safetensors")
    if not os.path.exists(latent_path):
        print(f"Generating latents for {video_path}...")
        cmd = [sys.executable, "scripts/util/video_to_latent.py", "--filelist", video_path]
        subprocess.run(cmd, env=ENV, cwd=ROOT_DIR)
    
    return landmark_path, latent_path

def run_inference(
    video_path, 
    audio_path, 
    keyframes_ckpt, 
    interpolation_ckpt, 
    what_mask, 
    resize_size, 
    strength,
    overlap,
    decoding_t,
    paste_back
):
    if not video_path or not audio_path:
        return "Please provide both video and audio files.", None
    
    # Preprocess
    try:
        preprocess_video(video_path)
    except Exception as e:
        return f"Preprocessing Error: {str(e)}", None

    # Checkpoints
    keyframes_ckpt_path = os.path.join(ROOT_DIR, "pretrained_models", "checkpoints", keyframes_ckpt) if keyframes_ckpt else "None"
    interpolation_ckpt_path = os.path.join(ROOT_DIR, "pretrained_models", "checkpoints", interpolation_ckpt) if interpolation_ckpt else "None"
    
    # Output name
    out_folder_name = f"gradio_{int(time.time())}"
    
    # Build command
    cmd = [
        sys.executable, "scripts/sampling/dubbing_pipeline.py",
        "--filelist", video_path,
        "--filelist_audio", audio_path,
        "--output_folder", out_folder_name,
        "--keyframes_ckpt", keyframes_ckpt_path,
        "--interpolation_ckpt", interpolation_ckpt_path,
        "--what_mask", what_mask,
        "--resize_size", str(resize_size),
        "--strength", str(strength),
        "--overlap", str(overlap),
        "--decoding_t", str(decoding_t),
        "--recompute", "True"
    ]
    if paste_back:
        cmd.append("--paste_back_to_body")
        cmd.append("True")
    
    try:
        process = subprocess.run(cmd, env=ENV, cwd=ROOT_DIR, capture_output=True, text=True)
        if process.returncode != 0:
            return f"Inference Error: {process.stderr}\n{process.stdout}", None
        
        # Look for the output video
        out_root = os.path.join(ROOT_DIR, "outputs")
        out_dir = os.path.join(out_root, out_folder_name)
        if os.path.exists(out_dir):
            files = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
            if files:
                return "Success!", os.path.join(out_dir, files[0])
        
        return f"Process finished but no output video found in {out_dir}. Logs: {process.stdout}", None
    except Exception as e:
        return f"Exception: {str(e)}", None

def start_training(
    filelist,
    batch_size,
    learning_rate,
    max_epochs,
    steps_per_ckpt,
    precision,
    accumulate_grad,
    base_config
):
    global training_process, training_logs_list
    if training_process and training_process.poll() is None:
        return "Training is already running!"
    
    training_logs_list = ["--- Initializing Training ---\n"]
    os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
    
    actual_config = base_config if (base_config and os.path.exists(base_config)) else "configs/example_training/keyframes/keyframes_dub.yaml"
    
    cmd = [
        sys.executable, "main.py",
        "--base", actual_config,
        "--train", "True",
        "data.params.train.datapipeline.filelist=" + filelist,
        "data.params.train.loader.batch_size=" + str(int(batch_size)),
        "model.base_learning_rate=" + str(learning_rate),
        "lightning.trainer.max_epochs=" + str(int(max_epochs)),
        "lightning.modelcheckpoint.params.every_n_train_steps=" + str(int(steps_per_ckpt)),
        "lightning.trainer.precision=" + precision,
        "lightning.trainer.accumulate_grad_batches=" + str(int(accumulate_grad)),
    ]
    
    # Start process
    training_process = subprocess.Popen(cmd, env=ENV, cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    # Start log reader thread
    thread = threading.Thread(target=log_reader, args=(training_process.stdout,))
    thread.daemon = True
    thread.start()
    
    return "Training started. Refresh status to see logs."

def stop_training():
    global training_process
    if training_process and training_process.poll() is None:
        training_process.terminate()
        return "Training stop signal sent."
    return "No training process running."

def get_training_logs():
    global training_logs_list, training_process
    logs = "".join(training_logs_list[-30:]) # Show last 30 lines
    if training_process:
        ret = training_process.poll()
        if ret is None:
            status = "\n[STATUS: RUNNING]"
        else:
            status = f"\n[STATUS: FINISHED (Code {ret})]"
    else:
        status = "\n[STATUS: IDLE]"
    return logs + status

# UI
with gr.Blocks(theme=gr.themes.Soft(), title="KeySync AI WebUI") as demo:
    gr.Markdown("# 🎭 KeySync AI: Lip-Sync & Video Generation")
    
    with gr.Tabs():
        with gr.Tab("🎬 Inference"):
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(label="Source Video (MP4)")
                    input_audio = gr.Audio(label="Target Audio (WAV/MP3)", type="filepath")
                    
                    with gr.Row():
                        kf_ckpt = gr.Dropdown(choices=get_checkpoints(), label="Keyframe Model")
                        interp_ckpt = gr.Dropdown(choices=get_checkpoints(), label="Interpolation Model")
                        refresh_ckpts = gr.Button("🔄", scale=0)
                    
                    with gr.Accordion("Generation Parameters", open=True):
                        full_body_stitch = gr.Checkbox(label="Full Body Stitching (Paste Back to Original Video)", value=True)
                        mask_type = gr.Dropdown(choices=["box", "full", "heart", "mouth"], value="box", 
                                              label="Mask Type ('box' or 'mouth' works best for stitching)")
                        res_size = gr.Slider(minimum=256, maximum=1024, step=256, value=512, label="Resolution")
                        inf_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=1.0, label="CFG Strength")
                        inf_overlap = gr.Slider(minimum=0, maximum=4, step=1, value=1, label="Overlap Frames")
                        inf_decoding_t = gr.Slider(minimum=1, maximum=25, step=1, value=14, label="Decoding Chunk (Lower if OOM)")
                    
                    btn_run = gr.Button("🚀 Start Generation", variant="primary")
                    
                with gr.Column():
                    output_status = gr.Textbox(label="Status Logs", lines=5)
                    output_video = gr.Video(label="Generated Video")
            
            refresh_ckpts.click(lambda: (gr.update(choices=get_checkpoints()), gr.update(choices=get_checkpoints())), outputs=[kf_ckpt, interp_ckpt])
            
            btn_run.click(
                run_inference, 
                inputs=[input_video, input_audio, kf_ckpt, interp_ckpt, mask_type, res_size, inf_strength, inf_overlap, inf_decoding_t, full_body_stitch],
                outputs=[output_status, output_video]
            )

        with gr.Tab("🏋️ Training"):
            # ... (Training tab remains the same)
            with gr.Row():
                with gr.Column():
                    train_filelist = gr.Textbox(label="Train Filelist Path (.txt)", placeholder="D:/datasets/train.txt")
                    train_config = gr.Textbox(label="Base YAML Config", value="configs/example_training/keyframes/keyframes_dub.yaml")
                    
                    with gr.Row():
                        train_batch_size = gr.Number(label="Batch Size (for 5090 try 4-8)", value=4)
                        train_acc_grad = gr.Number(label="Gradient Accumulation", value=1)
                    
                    with gr.Row():
                        train_lr = gr.Number(label="Learning Rate", value=1.e-5)
                        train_epochs = gr.Number(label="Max Epochs", value=1000)
                    
                    with gr.Row():
                        train_precision = gr.Dropdown(choices=["bf16-mixed", "32", "16-mixed"], value="bf16-mixed", label="Precision")
                        train_ckpt_freq = gr.Number(label="Save Frequency (steps)", value=5000)
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🚀 RTX 5090 Optimizations Applied")
                    gr.Markdown("- Recommended **Batch Size: 4-8**\n- Recommended **Precision: bf16-mixed**")
                    
                    with gr.Row():
                        btn_start = gr.Button("🔥 Start Training", variant="primary")
                        btn_stop = gr.Button("🛑 Stop Training", variant="stop")
                
                with gr.Column():
                    train_logs = gr.Textbox(label="Training Output (Real-time)", lines=15)
                    refresh_logs_btn = gr.Button("🔄 Refresh Status")
            
            btn_start.click(
                start_training,
                inputs=[train_filelist, train_batch_size, train_lr, train_epochs, train_ckpt_freq, train_precision, train_acc_grad, train_config],
                outputs=[train_logs]
            )
            btn_stop.click(stop_training, outputs=[train_logs])
            refresh_logs_btn.click(get_training_logs, outputs=[train_logs])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
