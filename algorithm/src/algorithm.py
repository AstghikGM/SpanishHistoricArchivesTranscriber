import base64
import os
from pathlib import Path

from ocean_runner import Algorithm
import requests

from src.data import Results

from docx import Document

from PIL import Image
import io

import time

OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")
OPENWEBUI_URL = "https://chat.agrospai.udl.cat"
CHAT_COMPLETIONS_URL = f"{OPENWEBUI_URL}/api/chat/completions"
MODEL_ID = "vllm.Qwen/Qwen3.6-27B-FP8"

VALID_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]
MAX_RETRIES = 3
PAUSE_BETWEEN_RETRIES = 60 #seconds
MAX_IMAGE_PX = 2800

algorithm = Algorithm()

    
def get_optimized_base64(path, max_px=MAX_IMAGE_PX):
    """Resizes and compresses image to prevent payload timeouts."""
    try:
        with Image.open(path) as img:
            w, h = img.size

            needs_resize = w > max_px or h > max_px
            if needs_resize:
                img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
                
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            buffer = io.BytesIO()
            final_quality = 85 if needs_resize else 95
            img.save(buffer, format="JPEG", quality=final_quality, optimize=True)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        algorithm.logger.error(f"Error processing image {path.name}: {e}")
        return None
    
def check_server_health(session):
    """Quick check on the Openwebui server by sending a test request."""
    algorithm.logger.info("Performing server health check...")
    data = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 5
    }

    try:
        response = session.post(CHAT_COMPLETIONS_URL, json=data, timeout=50)
        return response.status_code == 200
    except Exception as e:
        algorithm.logger.error(f"Health check failed: {e}")
        return False
         


@algorithm.run
def run(algorithm: Algorithm) -> Results:
    files = algorithm.job_details.files
    transcriptions = []

    if not OPENWEBUI_API_KEY:
        algorithm.logger.error("Missing API Key")
        return transcriptions
    
    with requests.Session() as session:
      session.headers.update({
          "Authorization": f"Bearer {OPENWEBUI_API_KEY}",
          "Content-Type": "application/json"
      })
      
      # Server health check
      if not check_server_health(session):
            algorithm.logger.error("Aborting: Server health check failed.")
            return transcriptions
      
      for file in files:
          for path in file.input_files:
              if path.suffix.lower() not in VALID_EXTENSIONS:
                  continue

              algorithm.logger.info(f"Adding file [{path.name}] for {file.did}")

              image_base64 = get_optimized_base64(path)
              if not image_base64:
                  algorithm.logger.error(f"Failed to optimize {path.name}")
                  continue
            
              system_prompt = "Handwritten text exact transcription."
              user_prompt = """
### ROLE: Professional Paleographer
### TASK: Diplomatic Transcription (20th-century SPANISH document image)

- **Verbatim:** Keep original spelling, grammar, and punctuation. No modernizing.
- **Layout:** Exact line-breaks. Start a NEW line for every line in the image.
- **Marginalia:** Transcribe all edge-text, signatures, and page numbers. Transcribe all the text.
- **Uncertainty:** Use ONLY the [word?] format to mark unclear text.
- **Output:** Output transcribed text ONLY. NO intro, NO comments, NO metadata description.
"""

              data = {
                  "model": MODEL_ID,
                  "stream": False,
                  "messages": [
                      {"role": "system", "content": system_prompt},
                      {
                          "role": "user",
                          "content": [
                              {"type": "text", "text": user_prompt},
                              {
                                  "type": "image_url",
                                  "image_url": {
                                      "url": f"data:image/jpeg;base64,{image_base64}"
                                  },
                              },
                          ],
                      }
                  ],
                  "temperature": 0,
                  "seed": 42,
                  "frequency_penalty": 0.0
              }
              
              success = False
              for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        response = session.post(CHAT_COMPLETIONS_URL, json=data, timeout=300)
                    
                        response.raise_for_status() 

                        response_json = response.json()

                        if not response_json or "choices" not in response_json or not response_json["choices"]:
                            algorithm.logger.debug(f"  [DEBUG] Json response: {response_json}")
                            raise ValueError("API response missing 'choices' or 'content'")
                       
                        content = response_json["choices"][0]["message"]["content"]
                        if not content or content.lower() == "none":
                            raise ValueError("API returned empty content")
                        
                        transcriptions.append({
                                "file_id": file.did,
                                "file_name": path.stem,
                                "content": content
                        })
                        success = True
                        break # Exit retry loop

                    except requests.exceptions.HTTPError as e:
                        status_code = e.response.status_code
                        if status_code in [429, 502, 503, 504]:
                            error_msg = f"Server temporarily busy (HTTP {status_code})"
                        else:
                            algorithm.logger.error(f" [!] Fatal HTTP Error {status_code}: {e.response.reason}")
                            break
                    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as e:
                        error_msg = f"Connection Error: {str(e)}"
                    except requests.exceptions.ReadTimeout:
                        error_msg = "Timeout: Server took too long to respond"
                    except Exception as e:
                        error_msg = f"Unexpected {type(e).__name__}: {str(e)}"
                  
                    # Exponential backoff
                    if attempt == MAX_RETRIES:
                        algorithm.logger.error(f" [!] Final failure for {path.name}: {error_msg}")
                    else:
                        wait_time = PAUSE_BETWEEN_RETRIES * attempt
                        algorithm.logger.warning(f" [!] Attempt {attempt} failed for {path.name}: {error_msg}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)           

              if not success:
                  algorithm.logger.error(f"CRITICAL: Failed to transcribe {path.name} after {MAX_RETRIES} attempts.")

    return transcriptions


@algorithm.save_results
def save(
    algorithm: Algorithm,
    result: Results,
    base: Path,
):
    for transcription in result:
        file_id = transcription["file_id"]
        file_name = transcription["file_name"]
        content = transcription["content"]

        output_dir = base / file_id
        output_dir.mkdir(parents=True, exist_ok=True)

        doc = Document()
        doc.add_paragraph(content)

        output_file = output_dir / f"{file_name}.docx"
        doc.save(str(output_file))

        algorithm.logger.info(f"Transcription saved in: {output_file}")



if __name__ == "__main__":
    algorithm()
