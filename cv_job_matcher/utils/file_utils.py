import os
import tempfile

async def save_upload_file(upload_file, destination_dir="uploads"):
    os.makedirs(destination_dir, exist_ok=True)
    suffix = os.path.splitext(upload_file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=destination_dir) as tmp:
        content = await upload_file.read()
        tmp.write(content)
        tmp_path = tmp.name
    return tmp_path
