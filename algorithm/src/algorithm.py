import base64
import os
import io
import time
from dotenv import get_key
from pathlib import Path
from typing import Any

import requests
from docx import Document
from PIL import Image

from pydantic import BaseModel, Field
from ocean_runner import Algorithm, Config


# OpenWebUi Configuration
OPENWEBUI_URL = "https://chat.agrospai.udl.cat"
CHAT_COMPLETIONS_URL = f"{OPENWEBUI_URL}/api/chat/completions"
VALID_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]
MAX_RETRIES = 3
PAUSE_BETWEEN_RETRIES = 60 #seconds
MAX_IMAGE_PX = 2800

# Inputs definition
class InputParameters(BaseModel):
    model_id: str = Field(
        default="Qwen/Qwen3.6-27b",
        title="Vision AI Model ID",
        description="The vision AI model to be used for the transcription."
    )

# Output Type Definition
type ResultsT = list[dict[str, Any]]

algorithm = Algorithm[InputParameters, ResultsT].create(
    Config(custom_input=InputParameters)
)

# Helper functions
def get_api_key() -> str:
    if algorithm.job_details.paths.algorithm.exists():
        api_key = get_key(str(algorithm.job_details.paths.algorithm), "OPENWEBUI_API_KEY")
    else:
        api_key = os.environ.get("OPENWEBUI_API_KEY")
 
    assert api_key is not None, (
        f"API key not populated from environment file {algorithm.job_details.paths.algorithm} nor environment"
    )
 
    return api_key

def _check_server_health(session: requests.Session, model_id: str) -> None:
    """Quick check on the OpenWebUI server."""
    algorithm.logger.info("Performing server health check...")
    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 5
    }
    
    try:
        health_res = session.post(
            CHAT_COMPLETIONS_URL, 
            json = data, 
            timeout=50
        )
        if health_res.status_code != 200:
            raise algorithm.Error("AI server is unreachable or returned an error.")
    except Exception as e:
        raise algorithm.Error(f"Health check failed: {str(e)}")
    
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
         
def call_llm_vision_api(session: requests.Session, system_prompt: str, user_prompt: str, image_base64: str, model_id: str) -> str:
    """HTTP requests, retries, and error handling adapted for Vision Models."""
    
    data = {
        "model": model_id,
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
        "frequency_penalty": 0.0,
        "chat_template_kwargs": {
        "enable_thinking": False
      }
    }
    
    # Main call with retries
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(CHAT_COMPLETIONS_URL, json=data, timeout=300)
            response.raise_for_status()
                    
            result_json = response.json()
            if not result_json or "choices" not in result_json:
                raise algorithm.Error("API response missing 'choices'.")
                    
            content = result_json['choices'][0]['message']['content']
            if not content or content.lower() == "none":
                raise algorithm.Error("API returned empty content.")
                    
            return content
                
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code in [429, 502, 503, 504]:
                error_msg = f"Server temporarily busy (HTTP {status_code})"
            else:
                algorithm.logger.error(f" [!] Fatal HTTP Error {status_code}: {e.response.reason}")
                raise algorithm.Error(f"LLM API failed with HTTP error {status_code}: {e.response.reason}")
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout) as e:
            error_msg = f"Connection Error: {str(e)}"
        except requests.exceptions.ReadTimeout:
            error_msg = "Timeout: Server took too long to respond"
        except Exception as e:
            error_msg = f"Unexpected {type(e).__name__}: {str(e)}"
                
        if attempt == MAX_RETRIES:
            raise algorithm.Error(f"Critical failure after {MAX_RETRIES} attempts. Reason: {error_msg}")
                
        wait_time = PAUSE_BETWEEN_RETRIES * attempt
        algorithm.logger.warning(f"Attempt {attempt} failed: {error_msg}. Retrying in {wait_time}s...")
        time.sleep(wait_time)


# Preconditions validation
@algorithm.validate
def validate(_) -> None:
    """Validation of preconditions."""
    algorithm.logger.info("Validating preconditions...")
    jd = algorithm.job_details
    assert jd.files, "The dataset does not contain any files to process."
    get_api_key() # Check credentials exist


# Main function
@algorithm.run
def run(_) -> ResultsT:
    """Main execution pipeline."""
    params = algorithm.job_details.input_parameters

    # Credentials
    api_key = get_api_key()

    # Prompts definition
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

    # Call Model API
    transcriptions = []

    with requests.Session() as session:
        session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        
        _check_server_health(session, params.model_id)

        # Process each file
        files = algorithm.job_details.files
        for file in files:
            for path in file.input_files:
                if path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                algorithm.logger.info(f"Adding file [{path.name}] for {file.did}")

                image_base64 = get_optimized_base64(path)
                if not image_base64:
                    algorithm.logger.error(f"Failed to optimize {path.name}")
                    continue

                try:
                    algorithm.logger.info(f"Calling vision model {params.model_id} for {path.name}...")
                    content = call_llm_vision_api(
                        session=session,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_base64=image_base64,
                        model_id=params.model_id
                    )
                    
                    transcriptions.append({
                        "file_id": file.did,
                        "file_name": path.stem,
                        "content": content
                    })
                    
                except algorithm.Error as e:
                    algorithm.logger.error(f"CRITICAL: Failed to transcribe {path.name} after {MAX_RETRIES} attempts. Skipping this file. Error: {e}")

    return transcriptions

# Save results
@algorithm.save_results
def save(_, result: ResultsT, base: Path):
    """Saving output files."""
    
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
