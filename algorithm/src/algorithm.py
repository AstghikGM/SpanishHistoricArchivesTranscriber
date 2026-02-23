import base64
import os
from pathlib import Path

from ocean_runner import Algorithm
import requests

from src.data import Results


from docx import Document

OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")

VALID_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]
OPENWEBUI_URL = "https://chat.agrospai.udl.cat"
CHAT_COMPLETIONS_URL = f"{OPENWEBUI_URL}/api/chat/completions"
#MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
MODEL_ID = "qwen3-vl:32b"


algorithm = Algorithm()


@algorithm.run
def run(algorithm: Algorithm) -> Results:
    files = algorithm.job_details.files

    transcriptions = []
    with requests.Session() as session:
      session.headers.update({
          "Authorization": f"Bearer {OPENWEBUI_API_KEY}",
          "Content-Type": "application/json"
      })

      for file in files:
          for path in file.input_files:
              if path.suffix.lower() not in VALID_EXTENSIONS:
                  continue

              algorithm.logger.info(f"Adding file [{path.name}] for {file.did}")

              with open(path, "rb") as f:
                  image_data = f.read()

              image_base64 = base64.b64encode(image_data).decode("utf-8")
              mime_type = {
                  ".jpg": "image/jpeg",
                  ".jpeg": "image/jpeg",
                  ".png": "image/png",
                  ".tiff": "image/tiff",
                  ".bmp": "image/bmp",
                  ".gif": "image/gif",
              }.get(path.suffix.lower(), "image/jpeg")

              prompt = """
Act as a professional paleographer. Transcribe literally this 20th-century historical Spanish document. 

Rules:
1. RAW LAYOUT: You must start a new line in your text every time a new line starts in the image. No exceptions.
2. VERBATIM ONLY: Transcribe text exactly as written (keep the spelling, archaic grammar, punctuation).
3. UNCERTAINTY: Use [?] for illegible words or [word?] if you are unsure of a word.
4. MARGINALIA: Transcribe all headers, page numbers, and signatures in the margins/corners.
5. FORMAT: Provide only the transcripted text. No additional explanations.
"""

              data = {
                  "model": MODEL_ID,
                  "stream": False,
                  "messages": [
                      {
                          "role": "user",
                          "content": [
                              {"type": "text", "text": prompt},
                              {
                                  "type": "image_url",
                                  "image_url": {
                                      "url": f"data:{mime_type};base64,{image_base64}"
                                  },
                              },
                          ],
                      }
                  ],
                  "temperature": 0,
                  "seed": 42,
                  "frequency_penalty": 0.0
              }

              try:
                   response = session.post(CHAT_COMPLETIONS_URL, json=data)
                   response.raise_for_status()
                 
                   response_json = response.json() 
                   if "choices" not in response_json:
                       algorithm.logger.error(response_json)
                       continue
                 
                   transcription = response_json["choices"][0]["message"]["content"]

                   transcriptions.append(
                       {
                           "file_id": file.did,
                           "file_name": path.stem,
                           "text": transcription
                       }
                   )

              except Exception as e:
                      algorithm.logger.error(f"Error in archive {path.name}: {e}")

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
        text = transcription["text"]

        output_dir = base / file_id
        output_dir.mkdir(parents=True, exist_ok=True)

        doc = Document()
        doc.add_paragraph(text)

        output_file = output_dir / f"{file_name}_TRY_3.docx"
        doc.save(str(output_file))

        algorithm.logger.info(f"Transcription saved in: {output_file}")



if __name__ == "__main__":
    algorithm()
