# Model configuration

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_MODEL = "Qwen/Qwen3-8B"
VISION_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

# RAG configuration

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# Video configuration

FRAME_INTERVAL = 10

# Folder configuration

DATA_DIR = "data"
FRAMES_DIR = "frames"
TRANSCRIPT_DB_DIR = "chroma_db"
VISUAL_DB_DIR = "visual_chroma_db"
RESULTS_DIR = "results"