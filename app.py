import os
import sys
import gc
import io
import tempfile
import base64
import torch
import numpy as np
import gradio as gr
from PIL import Image
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

# ── Constants ──────────────────────────────────────────────────────────────
EXP_DIR = os.path.join(BASE_DIR, "exp_outputs_mask")
REF_VIDEO_DIR = os.path.join(BASE_DIR, "benchmark_new", "reference_videos")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "..", "CogVideoX-5b-I2V")
MAX_OBJECTS = 5  # pipeline limit in common_inference.py

COLORS = [
    (255, 50, 50), (50, 255, 50), (50, 50, 255),
    (255, 255, 50), (255, 50, 255),
]


def get_available_motions():
    if not os.path.isdir(EXP_DIR):
        return []
    return sorted([
        d for d in os.listdir(EXP_DIR)
        if os.path.isfile(os.path.join(EXP_DIR, d, "pytorch_model.pt"))
    ])


def find_ref_video(motion_name):
    for category in ["animal", "human"]:
        path = os.path.join(REF_VIDEO_DIR, category, f"{motion_name}.mp4")
        if os.path.isfile(path):
            return path
    return None


# Cache converted videos so we only transcode once
_CONVERTED_VIDEO_DIR = os.path.join(BASE_DIR, ".ref_video_cache")
os.makedirs(_CONVERTED_VIDEO_DIR, exist_ok=True)


def get_browser_compatible_video(motion_name):
    """Return an HTML string with an animated GIF preview for any video format."""
    if not motion_name or motion_name == "-- select --":
        return ""

    # Check cache
    cached_path = os.path.join(_CONVERTED_VIDEO_DIR, f"{motion_name}.gif")
    if os.path.isfile(cached_path):
        with open(cached_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (
            f'<div style="text-align:center;">'
            f'<div style="margin-bottom:4px;font-weight:bold;color:#333;">Ref: {motion_name}</div>'
            f'<img src="data:image/gif;base64,{b64}" style="max-width:320px;border-radius:6px;display:block;margin:0 auto;"/>'
            f'</div>'
        )

    raw_path = find_ref_video(motion_name)
    if not raw_path:
        return f'<div style="text-align:center;color:#999;">No ref video for {motion_name}</div>'

    # Read frames, resize, save as GIF
    cap = cv2.VideoCapture(raw_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    scale = min(1.0, 320.0 / w)
    pil_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale < 1.0:
            new_w = int(frame.shape[1] * scale)
            new_h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(frame_rgb))
    cap.release()

    if not pil_frames:
        return f'<div style="text-align:center;color:#999;">Cannot read {motion_name}</div>'

    # Save GIF
    duration = int(1000 / fps)
    pil_frames[0].save(
        cached_path, save_all=True, append_images=pil_frames[1:],
        duration=duration, loop=0, optimize=True,
    )

    with open(cached_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    return (
        f'<div style="text-align:center;">'
        f'<div style="margin-bottom:4px;font-weight:bold;color:#333;">Ref: {motion_name}</div>'
        f'<img src="data:image/gif;base64,{b64}" style="max-width:320px;border-radius:6px;display:block;margin:0 auto;"/>'
        f'</div>'
    )


AVAILABLE_MOTIONS = get_available_motions()
MOTION_CHOICES = ["-- select --"] + AVAILABLE_MOTIONS


# ── Segmentation ──────────────────────────────────────────────────────────
def segment_objects(image, concept_words_str):
    if image is None:
        gr.Warning("Please upload a target image first.")
        return None, [], None

    concept_words = [w.strip() for w in concept_words_str.split("+") if w.strip()]
    if not concept_words:
        gr.Warning("Please enter concept/object words separated by '+'.")
        return None, [], None

    from tools.grounding_sam import grounded_segmentation
    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image

    try:
        image_array, detections = grounded_segmentation(
            image=image_pil, labels=concept_words,
            threshold=0.45, polygon_refinement=True,
        )
    except Exception as e:
        gr.Warning(f"Segmentation failed: {e}")
        return None, [], None

    sorted_dets = {label: [] for label in concept_words}
    for det in detections:
        label = det.label.rstrip('.')
        if label in sorted_dets:
            sorted_dets[label].append(det)

    ordered = []
    for label in concept_words:
        for det in sorted_dets.get(label, []):
            ordered.append((label, det))

    if not ordered:
        gr.Warning("No objects detected.")
        return None, [], None

    vis = image_array.astype(np.float32).copy()
    mask_list = []
    for i, (label, det) in enumerate(ordered):
        mask = det.mask
        color = COLORS[i % len(COLORS)]
        mask_bool = mask > 0
        for c in range(3):
            vis[:, :, c] = np.where(mask_bool, vis[:, :, c] * 0.5 + color[c] * 0.5, vis[:, :, c])
        mask_small = cv2.resize(mask, (45, 30), interpolation=cv2.INTER_NEAREST)
        mask_list.append(torch.from_numpy(mask_small.astype(bool)))

    vis = np.ascontiguousarray(vis.clip(0, 255).astype(np.uint8))
    for i, (label, det) in enumerate(ordered):
        color = COLORS[i % len(COLORS)]
        box = det.box
        cv2.rectangle(vis, (box.xmin, box.ymin), (box.xmax, box.ymax), color, 2)
        cv2.putText(vis, f"[{i+1}] {label}", (box.xmin, max(box.ymin - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    masks_tensor = torch.stack(mask_list)
    detected_labels = [label for label, _ in ordered]
    return vis, detected_labels, masks_tensor


# ── Inference ─────────────────────────────────────────────────────────────
def run_inference(
    image, prompt_text, concept_words_str, motion_words_str,
    masks_tensor, seed_val, guidance_scale, model_path,
    *selections,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Please upload a target image.")
    if masks_tensor is None:
        raise gr.Error("Please segment objects first.")

    motion_words = [w.strip() for w in motion_words_str.split("+") if w.strip()]
    num_objects = masks_tensor.shape[0]

    if len(motion_words) != num_objects:
        raise gr.Error(f"Need {num_objects} motion words, got {len(motion_words)}.")

    emb_ckpt_paths = []
    for i in range(num_objects):
        sel = selections[i] if i < len(selections) else None
        if not sel or sel == "-- select --":
            raise gr.Error(f"Please select a motion token for object {i+1}.")
        path = os.path.join(EXP_DIR, sel, "pytorch_model.pt")
        if not os.path.isfile(path):
            raise gr.Error(f"Checkpoint not found for '{sel}'.")
        emb_ckpt_paths.append(path)

    if not model_path or not os.path.isdir(model_path):
        raise gr.Error(f"CogVideoX model path invalid: {model_path}")

    from scripts.utils import get_gt_img
    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(tmp_fd)
    image_pil.save(tmp_path)

    try:
        gt_img = get_gt_img(tmp_path, 480, 720)
        output_dir = os.path.join(BASE_DIR, "gradio_outputs")
        os.makedirs(output_dir, exist_ok=True)
        save_name = f"gradio_seed{int(seed_val)}"

        from scripts.common_inference import generate_video
        video_path = generate_video(
            prompt=prompt_text,
            pretrained_model_name_or_path=model_path,
            emb_ckpt_paths=emb_ckpt_paths,
            gt_img=gt_img,
            gt_masks=masks_tensor,
            output_path=output_dir,
            guidance_scale=guidance_scale,
            seed=int(seed_val),
            high_timesteps=None,
            reweight_scale=None,
            motion_words=motion_words,
            save_name=save_name,
        )
    except gr.Error:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Inference failed: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return video_path


# ── Build Gradio UI ──────────────────────────────────────────────────────
def build_demo():
    with gr.Blocks(
        title="FlexiMMT - Multi-Object Multi-Motion Transfer",
        theme=gr.themes.Soft(),
    ) as demo:

        masks_state = gr.State(None)
        detected_labels_state = gr.State([])

        gr.Markdown("# FlexiMMT: Multi-Object Multi-Motion Transfer Demo")
        gr.Markdown(
            "Upload a target image, describe the scene, specify objects and motions, "
            "then assign trained motion tokens to transfer actions onto each object."
        )

        # ── Step 1 & 2 ───────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 1: Target Image & Prompt")
                target_image = gr.Image(label="Target Image", type="numpy")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g. A boy is doing squats. A dog stood up.",
                    lines=2,
                )
            with gr.Column(scale=1):
                gr.Markdown("### Step 2: Object & Motion Words")
                concept_words = gr.Textbox(
                    label="Object/Concept Words (separated by '+')",
                    placeholder="e.g. boy+dog",
                )
                motion_words_input = gr.Textbox(
                    label="Motion Words (separated by '+')",
                    placeholder="e.g. doing squats+stood up",
                )
                gr.Markdown(
                    "*Object words are used for segmentation. "
                    "Motion words must appear in the prompt and match the order of objects.*"
                )

        # ── Examples ──────────────────────────────────────────────────
        gr.Markdown("### Examples")
        gr.Markdown(
            "Click an example to auto-fill inputs, then click **Segment Objects** "
            "and select the suggested motion tokens."
        )
        examples_data = [
            [
                os.path.join(BASE_DIR, "assets/example/elephant1+movie_man_5.png"),
                "A elephant walks slowly on a grassland. A man dressed in a well is doing squats.",
                "elephant+man dressed in a well",
                "walks slowly+doing squats",
            ]
        ]
        examples_data = [e for e in examples_data if os.path.isfile(e[0])]
        if examples_data:
            gr.Examples(
                examples=examples_data,
                inputs=[target_image, prompt, concept_words, motion_words_input],
                label="Click to load an example",
                examples_per_page=5,
            )
            gr.Markdown(
                "**Suggested motion tokens:**\n"
                "Elephant + Man: `cows` → `crouch`"
            )

        # ── Step 3: Segmentation ──────────────────────────────────────
        gr.Markdown("### Step 3: Segment Objects")
        segment_btn = gr.Button("Segment Objects", variant="primary")
        mask_vis = gr.Image(label="Segmentation Result", interactive=False)

        # ── Step 4: Select Motion Tokens ──────────────────────────────
        gr.Markdown("### Step 4: Select Motion Tokens")
        gr.Markdown(
            "For each detected object, select a motion from the dropdown. "
            "The reference video will appear on the right. "
            "Multiple objects can share the same motion."
        )

        # Pre-create MAX_OBJECTS rows: [Dropdown | GIF preview]
        motion_rows = []
        motion_dropdowns = []
        ref_htmls = []

        for i in range(MAX_OBJECTS):
            row = gr.Row(visible=False)
            with row:
                with gr.Column(scale=1, min_width=250):
                    dd = gr.Dropdown(
                        choices=MOTION_CHOICES,
                        value="-- select --",
                        label=f"Object {i+1} Motion",
                        interactive=True,
                    )
                with gr.Column(scale=1, min_width=320):
                    rh = gr.HTML(value="")
            motion_rows.append(row)
            motion_dropdowns.append(dd)
            ref_htmls.append(rh)

        # ── Step 5: Parameters ────────────────────────────────────────
        gr.Markdown("### Step 5: Parameters & Run")
        with gr.Row():
            seed = gr.Number(label="Seed", value=42, precision=0)
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=15.0, value=6.0, step=0.5
            )
            model_path = gr.Textbox(
                label="CogVideoX-5b-I2V Model Path",
                value=DEFAULT_MODEL_PATH,
            )
        run_btn = gr.Button("Run Inference", variant="primary")

        # ── Output ────────────────────────────────────────────────────
        gr.Markdown("### Output Video")
        output_video = gr.Video(label="Generated Video")

        # ── Event: Segment ────────────────────────────────────────────
        def on_segment(image, concept_words_str):
            vis, labels, masks = segment_objects(image, concept_words_str)
            num = len(labels) if labels else 0

            results = [vis, labels, masks]
            for i in range(MAX_OBJECTS):
                if i < num:
                    results.append(gr.Row(visible=True))
                    results.append(gr.Dropdown(
                        choices=MOTION_CHOICES, value="-- select --",
                        label=f"[{i+1}] {labels[i]} → select motion",
                        interactive=True,
                    ))
                    results.append(gr.HTML(value=""))
                else:
                    results.append(gr.Row(visible=False))
                    results.append(gr.Dropdown(
                        choices=MOTION_CHOICES, value="-- select --",
                        label=f"Object {i+1} Motion", interactive=True,
                    ))
                    results.append(gr.HTML(value=""))
            return results

        segment_outputs = [mask_vis, detected_labels_state, masks_state]
        for i in range(MAX_OBJECTS):
            segment_outputs.extend([motion_rows[i], motion_dropdowns[i], ref_htmls[i]])

        segment_btn.click(
            fn=on_segment,
            inputs=[target_image, concept_words],
            outputs=segment_outputs,
        )

        # ── Event: Dropdown change -> show ref video as GIF ───────────
        def show_ref(motion_name):
            return get_browser_compatible_video(motion_name)

        for i in range(MAX_OBJECTS):
            motion_dropdowns[i].change(
                fn=show_ref,
                inputs=[motion_dropdowns[i]],
                outputs=[ref_htmls[i]],
            )

        # ── Event: Run Inference ──────────────────────────────────────
        run_btn.click(
            fn=run_inference,
            inputs=[
                target_image, prompt, concept_words, motion_words_input,
                masks_state, seed, guidance_scale, model_path,
                *motion_dropdowns,
            ],
            outputs=[output_video],
        )

    return demo


if __name__ == "__main__":
    # Clean up leftover tmp*.png files in project root from previous runs
    import glob
    for f in glob.glob(os.path.join(BASE_DIR, "tmp*.png")):
        try:
            os.remove(f)
        except OSError:
            pass

    demo = build_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
