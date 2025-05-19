import streamlit as st # type: ignore # type: ignore
import pandas as pd # type: ignore
from datetime import datetime
import requests # type: ignore
import os

FEEDBACK_FILE = "Feedback_Log.xlsx"

# === Session states ===
if "app_started" not in st.session_state:
    st.session_state.app_started = False
if "user_row_index" not in st.session_state:
    st.session_state.user_row_index = None
if "user_name" not in st.session_state:
    st.session_state.user_name = "Anonymous"

# === IP-based location fetch ===
def get_location():
    try:
        response = requests.get("https://ipinfo.io", timeout=5)
        data = response.json()
        return f"{data.get('city', 'Unknown')}, {data.get('country', 'Unknown')}"
    except:
        return "Unknown"

# === WELCOME SCREEN ===
if not st.session_state.app_started:
    st.image("Assets/logo.png", width=150)
    st.title("Welcome to SignBridge")

    st.markdown("""
    ### 🤝 Welcome to SignBridge

    Communication is a fundamental human right — yet millions in the deaf and mute community still face daily barriers. **SignBridge** is a first step toward building a more inclusive world.

    This MVP demonstrates how we can bridge communication between speech users and sign language users through real-time gesture recognition and speech translation.

    🎯 **What this tool offers:**
    - Convert **spoken phrases** into **sign gesture videos**
    - Recognize **dynamic hand gestures** and speak them aloud
    - Allow users to leave feedback and participate in refining the solution

    🌱 **This is just the beginning** — future versions aim to:
    - Use real-time webcam-based gesture recognition in live calls
    - Support a full dictionary of sign language phrases
    - Integrate with communication platforms and accessibility devices

    We invite you to explore the tool and experience how technology can truly bridge two communities.

    📝 Please try both modules and **share your valuable feedback** at the end. Every voice counts!

    Thank you for being part of this journey 🙏
    """)

    name_input = st.text_input("Enter your name (or leave blank to continue anonymously):")

    if st.button("🚀 Start Using SignBridge"):
        user_name = name_input.strip() if name_input.strip() else "Anonymous"
        st.session_state.user_name = user_name

        now = datetime.now()
        entry = {
            "User Name": user_name,
            "Date": now.strftime("%Y-%m-%d"),
            "Time": now.strftime("%H:%M:%S"),
            "Location": get_location(),
            "User Feedback": ""
        }

        if os.path.exists(FEEDBACK_FILE):
            df = pd.read_excel(FEEDBACK_FILE)
        else:
            df = pd.DataFrame(columns=["User Name", "Date", "Time", "Location", "User Feedback"])

        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        st.session_state.user_row_index = len(df) - 1
        df.to_excel(FEEDBACK_FILE, index=False)

        st.session_state.app_started = True
        st.rerun()

    st.stop()

# ========== MAIN TABS ==========
if st.session_state.app_started:
    tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Voice to Gesture", "✋ Gesture to Speech", "🗒️ Feedback", "🔐 Admin"])

    # ========== TAB 1 ==========
    with tab1:
        if st.session_state.app_started:
            import os
            import tempfile
            import speech_recognition as sr # type: ignore
            from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip, AudioFileClip, vfx # type: ignore
            from fuzzywuzzy import fuzz # type: ignore
            from gtts import gTTS # type: ignore

            # === Path Setup ===
            main_folder = r"C:\Users\srira\Desktop\SignBridge_Gestures\SignBridge_Dataset\Speech to Symbol\Display symbol\Words"
            output_folder = r"C:\Users\srira\Desktop\SignBridge_Gestures\SignBridge_Dataset\Speech to Symbol\Output"
            os.makedirs(output_folder, exist_ok=True)

            # === Alias Map ===
            ALIASES = {
                "shree": "sri", "shri": "sri", "siri": "sri", "tree": "sri",
                "tamizh": "tamil", "tamilnadu": "tamil_nadu", "namaste": "hello",
                "madras": "chennai", "delly": "delhi", "cheenai": "chennai",
                "tamilnad": "tamil_nadu", "tn": "tamil_nadu"
            }

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
                folder_list = os.listdir(main_folder)
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

                cleaned_text = " ".join([ALIASES.get(word, word) for word in spoken_text.lower().split()])
                st.info(f"🧠 Cleaned text: `{cleaned_text}`")

                clips = []
                for folder in matched_folders:
                    folder_path = os.path.join(main_folder, folder)
                    video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
                    if not video_files:
                        st.warning(f"⚠️ No video found in: `{folder}`")
                        continue
                    video_path = os.path.join(folder_path, video_files[0])
                    clip = VideoFileClip(video_path).fx(vfx.speedx, 1.25)

                    subtitle = TextClip(folder.replace("_", " "), fontsize=40, color='white', bg_color='black')
                    subtitle = subtitle.set_duration(clip.duration).set_position(lambda t: ('center', int(clip.h * 0.75)))
                    composite = CompositeVideoClip([clip, subtitle])
                    clips.append(composite)

                if not clips:
                    st.warning("⚠️ No valid clips to stitch.")
                    return None

                final_clip = concatenate_videoclips(clips)

                # === Add gTTS Audio ===
                tts = gTTS(text=cleaned_text, lang='en', slow=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    audio_path = fp.name
                final_audio = AudioFileClip(audio_path)
                final_clip = final_clip.set_audio(final_audio)

                output_filename = cleaned_text.lower().replace(" ", "_") + ".mp4"
                output_path = os.path.join(output_folder, output_filename)
                final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

                return output_path

            def transcribe_and_generate():
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    st.info("🎤 Listening... please speak now.")
                    audio = r.listen(source)
                try:
                    text = r.recognize_google(audio)
                    st.success(f"✅ You said: `{text}`")
                    
                    with st.spinner("⚙️ Generating gesture video, please wait..."):
                        output_path = stitch_and_generate(text)

                    if output_path:
                        st.success("✅ Gesture video created successfully!")
                        st.video(output_path)
                        st.markdown(f"📁 Saved to: `{output_path}`")

                except sr.UnknownValueError:
                    st.error("🙁 I couldn't hear or understand you. Please click the **Speak Now** button again and try speaking clearly.")
                except sr.RequestError as e:
                    st.error(f"❌ Network error: {e}")

            # === UI Section ===
            st.markdown("### 🎙️ Voice to Gesture")
            st.info("""
            🔔 This demo supports only specific phrases.
            Try examples like:
            - "Hi", "Hello", "How are you" 
            - "I love my work", "I love you"
            - "Yes", "No", "Thank you"
            """)

            if st.button("🎙️ Speak Now"):
                transcribe_and_generate()


    # ========== TAB 2 ==========
    with tab2:
        if st.session_state.app_started:
            import cv2 # type: ignore
            import numpy as np # type: ignore
            import mediapipe as mp # type: ignore
            from tensorflow.keras.models import load_model # type: ignore
            from gtts import gTTS # type: ignore
            import tempfile
            import pygame # type: ignore
            import time
            import os

            # === Load model and class labels ===
            MODEL_PATH = r"C:\Users\srira\Desktop\SignBridge_Gestures\Model\gesture_model.h5"
            model = load_model(MODEL_PATH)
            CLASSES = ['Hello', 'I', 'From', 'Tamil_Nadu', 'Thank_You']

            # === Mediapipe setup ===
            mp_holistic = mp.solutions.holistic
            mp_drawing = mp.solutions.drawing_utils

            # === Utils ===
            def extract_keypoints(results):
                pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
                return np.concatenate([pose, lh, rh])

            def reshape_landmark_vector(vector, target_size=126):
                if vector.shape[0] == target_size:
                    return vector
                elif vector.shape[0] > target_size:
                    return vector[:target_size]
                else:
                    return np.concatenate([vector, np.zeros(target_size - vector.shape[0])])

            # === UI Header ===
            st.markdown("### ✋ Gesture to Speech")
            st.info("Perform one of the following gestures: `Hello`, `I`, `From`, `Tamil_Nadu`, `Thank_You`")

            # === Start Prediction Button ===
            if st.button("▶️ Start 10-sec Gesture Prediction"):
                st.warning("🟡 Gesture prediction started. Please show your sign...")

                cap = cv2.VideoCapture(0)
                sequence = []
                prev_pred = ''
                start_time = time.time()

                frame_placeholder = st.empty()
                prediction_placeholder = st.empty()

                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    while time.time() - start_time < 10:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        image = cv2.resize(frame, (640, 480))
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = holistic.process(image_rgb)

                        keypoints = extract_keypoints(results)
                        keypoints = reshape_landmark_vector(keypoints)
                        sequence.append(keypoints)

                        # Display live frame in Streamlit
                        frame_placeholder.image(image_rgb, channels="RGB")

                        if len(sequence) == 30:
                            input_seq = np.expand_dims(sequence, axis=0)
                            res = model.predict(input_seq)[0]
                            prediction = CLASSES[np.argmax(res)]
                            confidence = res[np.argmax(res)]

                            if confidence > 0.8 and prediction != prev_pred:
                                prev_pred = prediction
                                prediction_placeholder.success(f"🧠 Prediction: **{prediction}** ({confidence*100:.2f}%)")

                                # gTTS Speak (safe cross-platform fix)
                                tts = gTTS(text=prediction.replace("_", " "), lang='en')
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                                    tts.save(tf.name)
                                    temp_path = tf.name

                                time.sleep(0.5)  # ensure OS finishes writing before playing

                                pygame.mixer.init()
                                pygame.mixer.music.load(temp_path)
                                pygame.mixer.music.play()
                                while pygame.mixer.music.get_busy():
                                    time.sleep(0.1)

                                try:
                                    os.remove(temp_path)
                                except:
                                    pass  # ignore if Windows locks the file
                                                # === Cleanup webcam properly ===
                cap.release()
                del cap
                cv2.destroyAllWindows()
                st.success("✅ Prediction finished and camera released.")


    # ========== TAB 3 ==========
    with tab3:
        if st.session_state.app_started:
            st.markdown("### 🗒️ Your Feedback")
            st.info("Please share how your experience was. Your voice matters!")

            feedback = st.text_area("Your Feedback")

            if st.button("📥 Submit Feedback"):
                if not feedback.strip():
                    st.warning("⚠️ Feedback cannot be empty.")
                else:
                    if os.path.exists(FEEDBACK_FILE):
                        df = pd.read_excel(FEEDBACK_FILE)

                        # Update the user's feedback in their session row
                        if st.session_state.user_row_index is not None:
                            df.at[st.session_state.user_row_index, "User Feedback"] = feedback.strip()
                            df.to_excel(FEEDBACK_FILE, index=False)
                            st.success("✅ Thank you! Your feedback has been saved.")
                        else:
                            st.error("User session not found.")
                    else:
                        st.error("Feedback log file missing.")

    # ========== TAB 4 ==========
    with tab4:
        st.markdown("### 🔐 Admin Panel - Feedback Log Access")
        admin_pass = st.text_input("Enter admin password", type="password")

        if admin_pass == "signadmin123":
            st.success("✅ Access granted. Welcome, Admin!")

            try:
                df = pd.read_excel(FEEDBACK_FILE)

                st.markdown("### 📋 All Feedback Records")
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    label="📥 Download Feedback as CSV",
                    data=df.to_csv(index=False),
                    file_name="Feedback_Log.csv",
                    mime='text/csv'
                )
            except FileNotFoundError:
                st.warning("⚠️ No feedback log found yet.")
        elif admin_pass:
            st.error("❌ Incorrect password.")

