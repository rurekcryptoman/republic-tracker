import os
import time
import tempfile
from dotenv import load_dotenv
from PIL import Image
from supabase import create_client
from diffusers import StableDiffusionImg2ImgPipeline
import torch

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
AI_BUCKET = os.getenv("AI_BUCKET", "ai-images")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "8"))
DEVICE = os.getenv("DEVICE", "cpu")

MODEL_ID = "nitrosocke/Ghibli-Diffusion"

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

print("Connecting to Supabase...")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

print("Loading Ghibli diffusion model...")

if DEVICE == "cuda":
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
    )
else:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32
    )

pipe = pipe.to(DEVICE)

if DEVICE == "cuda":
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

print(f"Model loaded on {DEVICE}")


def stylize_image(input_path: str, output_path: str):
    init_image = Image.open(input_path).convert("RGB")
    init_image = init_image.resize((512, 512))

    prompt = (
        "ghibli style portrait, soft anime illustration, beautiful face, "
        "clean line art, expressive eyes, soft pastel colors, detailed hair, "
        "gentle shading, cozy clothing, cinematic light, high quality"
    )

    negative_prompt = (
        "realistic photo, 3d, blurry, low quality, noisy, ugly, distorted face, "
        "bad anatomy, extra fingers, deformed hands, harsh shadows"
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=0.72,
        guidance_scale=8.5,
        num_inference_steps=35
    ).images[0]

    result.save(output_path, format="PNG")


def with_retry(fn, label, retries=6, delay=3):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            print(f"[RETRY] {label} failed (attempt {attempt}/{retries}) -> {type(e).__name__}: {e}")
            time.sleep(delay)
    raise last_error


def download_storage_file(path_in_bucket: str, local_path: str):
    def _download():
        return supabase.storage.from_(AI_BUCKET).download(path_in_bucket)

    file_bytes = with_retry(_download, f"download {path_in_bucket}")

    with open(local_path, "wb") as f:
        f.write(file_bytes)


def upload_storage_file(local_path: str, path_in_bucket: str):
    def _upload():
        with open(local_path, "rb") as f:
            return supabase.storage.from_(AI_BUCKET).upload(
                path_in_bucket,
                f,
                {"content-type": "image/png", "upsert": "true"}
            )

    return with_retry(_upload, f"upload {path_in_bucket}")


def update_job(job_id: str, values: dict):
    def _update():
        return supabase.table("ai_jobs").update(values).eq("id", job_id).execute()

    return with_retry(_update, f"update job {job_id}")


def get_next_job():
    def _get():
        return (
            supabase.table("ai_jobs")
            .select("*")
            .eq("status", "pending")
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )

    response = with_retry(_get, "get_next_job")
    data = response.data or []
    return data[0] if data else None


def process_job(job: dict):
    job_id = job["id"]
    input_image = job.get("input_image")

    if not input_image:
        update_job(job_id, {"status": "failed"})
        print(f"[FAILED] {job_id} missing input image")
        return

    print(f"[START] Processing job {job_id}")

    try:
        update_job(job_id, {"status": "processing"})
    except Exception as e:
        print(f"[ERROR] could not mark job as processing -> {type(e).__name__}: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        input_local = os.path.join(tmpdir, "input.png")
        output_local = os.path.join(tmpdir, "output.png")

        try:
            print(f"[{job_id}] Downloading input: {input_image}")
            download_storage_file(input_image, input_local)

            print(f"[{job_id}] Running stylize_image()")
            stylize_image(input_local, output_local)

            if not os.path.exists(output_local):
                raise RuntimeError("Output image was not created")

            output_path = f"output/{job_id}.png"
            print(f"[{job_id}] Uploading output: {output_path}")
            upload_storage_file(output_local, output_path)

            print(f"[{job_id}] Updating ai_jobs row")
            update_job(
                job_id,
                {
                    "status": "completed",
                    "output_image": output_path
                }
            )

            print(f"[DONE] {job_id} -> {output_path}")

        except Exception as e:
            try:
                update_job(job_id, {"status": "failed"})
            except Exception as update_error:
                print(f"[ERROR] failed to mark job as failed -> {type(update_error).__name__}: {update_error}")

            print(f"[ERROR] {job_id} -> {type(e).__name__}: {e}")


def main():
    print("AI Worker started")
    print(f"Polling every {POLL_INTERVAL}s")

    while True:
        try:
            job = get_next_job()
            if job:
                process_job(job)
            else:
                print("No pending jobs")
        except Exception as e:
            print(f"[LOOP ERROR] {type(e).__name__}: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()