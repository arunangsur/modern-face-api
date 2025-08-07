# =================================================================
#       main.py (Modern, Lightweight, using DeepFace)
# =================================================================

import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from deepface import DeepFace
import shutil

# --- 1. Initialize the FastAPI Application ---
app = FastAPI(title="Modern & Lightweight Face Recognition API")

# --- 2. Setup the Image Database Path ---
# DeepFace works by having a folder of images. Each subfolder is a person's identity.
# e.g., db_path/STU2025101/photo.jpg
DATA_DIR = os.environ.get('RENDER_DISK_PATH', 'face_db')
DB_PATH = DATA_DIR

os.makedirs(DB_PATH, exist_ok=True)


# --- 3. Define API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Modern Face Recognition API is running."}


@app.post("/register")
async def register_face(user_id: str = Form(...), file: UploadFile = File(...)):
    """
    Registers a new face by saving their image into a dedicated folder.
    """
    # Create a folder for the user if it doesn't exist
    user_folder_path = os.path.join(DB_PATH, user_id)
    os.makedirs(user_folder_path, exist_ok=True)
    
    # Save the user's photo. We'll name it 'face.jpg'.
    # This will overwrite any existing photo, allowing for updates.
    file_path = os.path.join(user_folder_path, "face.jpg")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # DeepFace automatically updates its index, but we can delete the pickle file
    # to force a re-index, which is safer.
    if os.path.exists(os.path.join(DB_PATH, "representations_vgg_face.pkl")):
        os.remove(os.path.join(DB_PATH, "representations_vgg_face.pkl"))
        
    print(f"Successfully registered/updated face for user: {user_id}")
    return {"status": "success", "user_id": user_id}


@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    """
    Identifies a face from an image by searching the image database.
    """
    # Save the uploaded file to a temporary path to be processed
    temp_file_path = "temp_identify_face.jpg"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # --- The Core DeepFace Logic ---
        # Find the closest match in our database folder.
        # We use a smaller, faster model 'VGG-Face' and 'cosine' distance.
        # The enforce_detection=False flag prevents crashes if no face is in the image.
        dfs = DeepFace.find(
            img_path=temp_file_path, 
            db_path=DB_PATH, 
            model_name="VGG-Face",
            distance_metric="cosine",
            enforce_detection=False
        )

        # DeepFace.find returns a list of pandas DataFrames. We care about the first one.
        if not dfs or dfs[0].empty:
            print("No matching face found in the database.")
            return {"status": "no_match_found"}

        # The 'identity' column contains the path to the matched image.
        # e.g., 'face_db/STU2025101/face.jpg'
        identity_path = dfs[0].iloc[0].identity
        
        # We extract the user_id, which is the name of the folder.
        user_id = os.path.basename(os.path.dirname(identity_path))
        
        print(f"Match found: {user_id}")
        return {"status": "match_found", "user_id": user_id}

    except Exception as e:
        print(f"An error occurred during identification: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)