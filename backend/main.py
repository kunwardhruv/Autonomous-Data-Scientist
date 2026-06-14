# main.py — Project entry point
#
# Run karne ke liye: python main.py
# Ya directly: uvicorn api.server:app --reload

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",   # All interfaces par listen karo
        port=8000,
        reload=True,       # Code change hone par auto-restart (development mode)
    )
