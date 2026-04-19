# Plan: Port Emotion Detector to Streamlit

## 1. Setup Streamlit
- Add `streamlit-webrtc` to requirements
- Create new file `streamlit_app.py`

## 2. Replace GUI Framework
- Remove customtkinter
- Replace with Streamlit:
  - `st.title()` for title
  - `st.button()` for 8 model selection buttons (2 rows x 4 cols - exactly like app.py)
  - Layout: Final Main | Final Stats (top row, centered), Hybrid Main | Hybrid Stats | ML Main | ML Stats | Geo Main | Geo Stats (bottom row)

## 3. Handle Camera
- Use `streamlit-webrtc` for real-time camera feed
- Process each frame through callback

## 4. Preserve All Model Logic
- Copy all 8 model implementations from app.py:
  1. ML Main
  2. ML Stats
  3. Geo Main
  4. Geo Stats
  5. Hybrid Main
  6. Hybrid Stats
  7. Final Main
  8. Final Stats

## 5. State Management
- Use `st.session_state` for:
  - Current selected model
  - Buffers (ai_buffer, geo_buffer, hybrid_buffer)
  - Video processor state

## 6. Layout
- Exactly like app.py:
  - Top: Title
  - Buttons: 2x1 top row (Final Main, Final Stats centered), 2x3 bottom row
  - Middle: Live camera feed
  - Bottom: Photo button + capture to Pictures folder
