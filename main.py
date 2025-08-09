# =================================================================
#   main.py (Production-Ready for Render with In-Memory Processing)
# =================================================================

import os
import uvicorn
import numpy as np
import pickle
import io
import shutil
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Dict, Any
from deepface import DeepFace

# --- 1. Initialize FastAPI Application ---
app = FastAPI(title="Production Face Recognition API")

# --- 2. Setup Persistent Storage on Render ---
# We MUST use the persistent disk for our image database.
DB_PATH = os.environ.get('RENDER_DISK_PATH', 'face_db_local')
os.makedirs(DB_PATH, exist_ok=True)
print(f"✅ Database path is set to: {DB_PATH}")

# --- 3. Pre-load a Model to Speed Up First Request (Optional but good) ---
try:
    print("⏳ Pre-loading face recognition model...")
    _ = DeepFace.build_model("SFace")
    print("✅ Model pre-loaded successfully.")
except Exception as e:
    print(f"🔥 Could not pre-load model: {e}")

# --- 4. Define API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Face Recognition API is running and healthy."}

@app.post("/register")
async def register_face(user_id: str = Form(...), file: UploadFile = File(...)):
    """
    Registers a new face by saving their image to a dedicated folder on the persistent disk.
    """
    try:
        # Create a unique folder for the user on the persistent disk
        user_folder_path = os.path.join(DB_PATH, user_id)
        os.makedirs(user_folder_path, exist_ok=True)
        
        # Save the user's photo. This will overwrite any existing photo for updates.
        file_path = os.path.join(user_folder_path, "face.jpg")
        
        # Read file bytes and write to the persistent disk
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
            
        # IMPORTANT: Force DeepFace to re-index its database.
        # This deletes the pickle file so it has to scan the folders again.
        representations_path = os.path.join(DB_PATH, "representations_sface.pkl")
        if os.path.exists(representations_path):
            os.remove(representations_path)
            print(f"Deleted old index file at {representations_path} to force re-indexing.")
            
        print(f"Successfully registered/updated face for user: {user_id}")
        return {"status": "success", "user_id": user_id}

    except Exception as e:
        print(f"🔥 Error during registration for {user_id}: {e}")
        # Return a proper server error
        raise HTTPException(status_code=500, detail=f"An internal error occurred during registration: {e}")


@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    """
    Identifies a face from an image ENTIRELY IN-MEMORY.
    """
    try:
        # Read the uploaded image into memory as bytes
        image_bytes = await file.read()
        
        # Convert the bytes to a numpy array that DeepFace can use directly
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)

        # --- The Core In-Memory DeepFace Logic ---
        # We pass the numpy array directly to the `img_path` parameter.
        # DeepFace is smart enough to handle it without a temp file.
        dfs = DeepFace.find(
            img_path = image_np,
            db_path = DB_PATH, 
            model_name = "SFace",
            distance_metric = "cosine",
            enforce_detection = False
        )
        
        if not dfs or dfs[0].empty:
            print("No matching face found in the database.")
            # Return a clear "not found" status
            raise HTTPException(status_code=404, detail="No matching face was found in the database.")

        # Extract the user_id from the identity path
        identity_path = dfs[0].iloc[0].identity
        user_id = os.path.basename(os.path.dirname(identity_path))
        
        print(f"✅ Match found: {user_id}")
        # Note: DeepFace doesn't easily return the location when using find().
        # We are simplifying to just return the user_id for now to ensure stability.
        return {"status": "match_found", "user_id": user_id}

    except HTTPException as e:
        # Re-raise HTTP exceptions so FastAPI handles them correctly
        raise e
    except Exception as e:
        print(f"🔥 An error occurred during identification: {e}")
        # Return a proper server error for all other exceptions
        raise HTTPException(status_code=500, detail=f"An internal error occurred during identification: {e}")

# The `if __name__ == "__main__":` block is removed for deployment.
