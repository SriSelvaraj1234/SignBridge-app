# Web-Ready SignBridge Streamlit App (Revised for Deployment)

import streamlit as st
import pandas as pd
from datetime import datetime
import os
from fuzzywuzzy import fuzz
from gtts import gTTS
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip, AudioFileClip, vfx
import tempfile

# === Session States ===
if "app_started" not in st.session_state:
    st.session_state.app_started = False
if "user_name" not in st.session_state:
    st.session_state.user_name = "Anonymous"
if "feedback_list" not in st.session_state:
    st.session_state.feedback_list = []

# === Constants ===
MAIN_FOLDER = "GestureVideos"
OUTPUT_FOLDER = "GestureVideos/Output"
LOGO_PATH = "Assets/logo.png"
ALIASES = {
    "shree": "sri", "shri": "sri", "siri": "sri", "tree": "sri",
    "tamizh": "tamil", "tamilnadu": "tamil_nadu", "namaste": "hello",
    "madras": "chennai", "delly": "delhi", "cheenai": "chennai",
    "tamilnad": "tamil_nadu", "tn": "tamil_nadu"
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_best_match(word, folder_list, threshold=85):
    best_match = ""
    highest_score = 0
    for folder in folder_list:
        score = fuzz.ratio(word, folder.lower())
        if score > highest_score and score >= threshold:
            highest_score = score
            best_match = folder
    return best_match

def stitch_and_generate(spoken_text):
    folder_list = os.listdir(MAIN_FOLDER)
    raw_words = spoken_text.lower().split()
    word_list = [ALIASES.get(word, word) for word in raw_words]

    matched_folders = []
    skip_next = False
    for i in range(len(word_list)):
        if skip_next:
            skip_next = False
            continue
        two_word_phrase = "_".join(word_list[i:i+2])
        match = get_best_match(two_word_phrase, folder_list)
        if match:
            matched_folders.append(match)
            skip_next = True
        else:
            match = get_best_match(word_list[i], folder_list)
            if match:
                matched_folders.append(match)

    if not matched_folders:
        st.warning("❌ No matching folders found.")
        return None

    cleaned_text = " ".join(word_list)
    st.info(f"🧠 Cleaned text: `{cleaned_text}`")

    clips = []
    for folder in matched_folders:
        folder_path = os.path.join(MAIN_FOLDER, folder)
        video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
        if not video_files:
            st.warning(f"⚠️ No video found in: `{folder}`")
            continue
        video_path = os.path.join(folder_path, video_files[0])
        clip = VideoFileClip(video_path).fx(vfx.speedx, 1.25)
        subtitle = TextClip(folder.replace("_", " "), fontsize=40, color='white', bg_color='black')
        subtitle = subtitle.set_duration(clip.duration).set_position(lambda t: ('center', int(clip.h * 0.75)))
        clips.append(CompositeVideoClip([clip, subtitle]))

    if not clips:
        return None

    final_clip = concatenate_videoclips(clips)

    # gTTS
    tts = gTTS(text=cleaned_text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
        tts.save(tf.name)
        audio_path = tf.name
    final_audio = AudioFileClip(audio_path)
    final_clip = final_clip.set_audio(final_audio)

    output_filename = cleaned_text.lower().replace(" ", "_") + ".mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path

# === WELCOME SCREEN ===
if not st.session_state.app_started:
    st.image(LOGO_PATH, width=150)
    st.title("Welcome to SignBridge")
    name_input = st.text_input("Enter your name (or leave blank to continue anonymously):")
    if st.button("🚀 Start Using SignBridge"):
        st.session_state.user_name = name_input.strip() or "Anonymous"
        st.session_state.app_started = True
        st.rerun()
    st.stop()

# === MAIN TABS ===
tab1, tab2, tab3 = st.tabs(["🎙️ Voice to Gesture", "✋ Gesture to Speech", "🗒️ Feedback"])

with tab1:
    st.markdown("### 🎙️ Voice to Gesture")
    spoken_text = st.text_input("🗣️ Type a phrase to convert into gesture video")
    if st.button("🎬 Generate Video") and spoken_text.strip():
        with st.spinner("Generating gesture video..."):
            result_path = stitch_and_generate(spoken_text.strip())
            if result_path:
                st.success("✅ Gesture video created!")
                st.video(result_path)
                st.markdown(f"📁 Saved to: `{result_path}`")

with tab2:
    st.markdown("### ✋ Gesture to Speech")
    st.warning("⚠️ This feature uses webcam and works only in the desktop version of this app.")

with tab3:
    st.markdown("### 🗒️ Your Feedback")
    feedback = st.text_area("Share your experience with us:")
    if st.button("📥 Submit Feedback"):
        if feedback.strip():
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.feedback_list.append({
                "User": st.session_state.user_name,
                "Time": now,
                "Feedback": feedback.strip()
            })
            st.success("✅ Feedback recorded. Thank you!")
        else:
            st.warning("⚠️ Feedback cannot be empty.")

    if st.session_state.feedback_list:
        st.markdown("### 🧾 Submitted Feedback")
        st.dataframe(pd.DataFrame(st.session_state.feedback_list))

