import streamlit as st
import tempfile
import pandas as pd
import librosa
import subprocess
import numpy as np
from essentia.standard import MonoLoader
from maest import get_maest
from streamlit_option_menu import option_menu
import uuid

@st.cache_resource
def load_model():
    model = get_maest(arch="discogs-maest-30s-pw-129e-519l")
    model.eval()
    return model

model = load_model()

# taxonomy definition
GENRE_THRESHOLDS = {'Disco': 0.23, 'Drum n Bass': 0.8, 'Hard Techno': 0.9, 'Progressive House': 0.47, 'Psy-Trance': 0.64, 'Techno': 0.74, 'Trance': 0.45}
ALLOWED_GENRES = list(GENRE_THRESHOLDS.keys())

# audio preprocessing
def find_loudest_segment(audio_path, target_duration=30.0):
    try:
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return 0.0
    hop_length = 512
    frame_duration = hop_length / sr
    frames_per_window = int(target_duration / frame_duration)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    window_energies = np.convolve(rms, np.ones(frames_per_window), mode='valid')
    if len(window_energies) == 0:
        return 0.0
    best_frame = np.argmax(window_energies)
    return best_frame * frame_duration

def convert_audio_to_30s_segment(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_path = tmp_in.name
    output_path = input_path.replace(".mp3", "_processed.wav").replace(".wav", "_processed.wav")
    start_time = find_loudest_segment(input_path)
    subprocess.run([
        "ffmpeg", "-ss", str(start_time), "-i", input_path,
        "-t", "30", "-ac", "1", "-ar", "16000", output_path,
        "-y", "-loglevel", "error"
    ])
    return output_path

# sidebar
with st.sidebar:
    st.markdown("# Automated Music Tagging for EDM Track Filtering")
    st.markdown("### Select your role")
    role = option_menu(
        menu_title=None,
        options=["Curator", "Producer"],
        icons=["sliders", "music-note"],
        menu_icon=None,
        default_index=0,
        orientation="vertical"
    )

    if "drops" not in st.session_state:
        st.session_state.drops = []

    if "submissions" not in st.session_state:
        st.session_state.submissions = []

# curator
if role == "Curator":

    tab1, tab2, tab3 = st.tabs(["Create New Drop", "Edit Existing Drop", "Submissions Overview"])
    
    with tab1:
        st.markdown("# Create New Drop")

        # name drop
        st.markdown("#### Drop Name")
        drop_name = st.text_input("", placeholder="Enter a name for this drop", label_visibility="collapsed")

        # drop type
        st.markdown("#### Drop Type")
        drop_type = st.radio(
            label="", 
            options=["Demo", "Promo"], 
            horizontal=True,
            label_visibility="collapsed"
        )

        # genre selection
        st.markdown("#### Select Genres")
        selected_genres = st.multiselect("", ALLOWED_GENRES, label_visibility="collapsed")

        # slider
        st.markdown("#### Recall vs Precision")
        st.markdown("Left = Lower Threshold (More Recall), Right = Higher Threshold (More Precision)")
        slider_pos = st.slider(
            label="",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            format="",
            label_visibility="collapsed"
        )
        st.markdown(f"**Preference: {'Less Genre Precision' if slider_pos <= 0.33 else 'More Genre Precision' if slider_pos >= 0.67 else 'Balanced'}**")

        # drop creation button
        button_style = """
        <style>
        div.stButton > button:first-child {
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            padding: 0.6em 1.5em;
            border-radius: 10px;
            border: none;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            transition: 0.3s;
        }
        div.stButton > button:first-child:hover {
            background-color: #ff1c1c;
            transform: scale(1.02);
        }
        </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)

        if st.button("Create Drop"):
            drop_id = str(uuid.uuid4())[:8]
            thresholds = {}
            for genre in selected_genres:
                base_th = GENRE_THRESHOLDS[genre]
                adjustment = 1.25 - 0.5 * slider_pos
                thresholds[genre] = round(base_th * adjustment, 2)

            drop = {
                "id": drop_id,
                "name": drop_name,
                "type": drop_type,
                "genres": selected_genres,
                "thresholds": thresholds
            }

            st.session_state.drops.append(drop)
            st.success(f"Drop '{drop_name}' created!")

    # edit drop
    with tab2:
        st.markdown("# Edit Existing Drop")

        # drop selection
        if not st.session_state.drops:
            st.warning("No drops to edit yet.")
        else:
            st.markdown("#### Select a Drop to Edit")
            drop_to_edit = st.selectbox(
                label="",
                options=[f"{drop['type'].title()} Drop: {drop['name']}" for drop in st.session_state.drops],
                label_visibility="collapsed"
            )

            index = next(i for i, d in enumerate(st.session_state.drops)
                            if f"{d['type'].title()} Drop: {d['name']}" == drop_to_edit)
            drop = st.session_state.drops[index]

            # edit
            st.markdown("#### Edit Drop Name")
            new_name = st.text_input("", drop["name"], label_visibility="collapsed", key="edit_name")

            st.markdown("#### Edit Drop Type")
            new_type = st.radio(
                label="",
                options=["Demo", "Promo"],
                index=["Demo", "Promo"].index(drop["type"]),
                horizontal=True,
                label_visibility="collapsed",
                key="edit_type"
            )

            st.markdown("#### Edit Genres")
            new_genres = st.multiselect("", ALLOWED_GENRES, default=drop["genres"], label_visibility="collapsed", key="edit_genres")

            st.markdown("#### Adjust Recall vs Precision")
            st.markdown("Left = Lower Threshold (More Recall), Right = Higher Threshold (More Precision)")
            new_slider = st.slider(
                label="",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                format="",
                label_visibility="collapsed",
                key="edit_slider"
            )
            st.markdown(f"**Preference: {'Less Genre Precision' if new_slider <= 0.33 else 'More Genre Precision' if new_slider >= 0.67 else 'Balanced'}**")

            # save changes button
            if st.button("Save Changes"):
                adjustment = 1.25 - 0.5 * new_slider
                thresholds = {genre: round(GENRE_THRESHOLDS[genre] * adjustment, 2) for genre in new_genres}

                st.session_state.drops[index] = {
                    "id": drop["id"],
                    "name": new_name,
                    "type": new_type,
                    "genres": new_genres,
                    "thresholds": thresholds
                }
                st.success("Drop updated successfully!")

            # delete drop button
            if st.button("Delete Drop"):
                st.session_state.drops.pop(index)
                st.success("Drop deleted.")
    
    # submissions overview
    with tab3:
        st.markdown("# Submissions Overview")

        if not st.session_state.drops:
            st.warning("No drops created yet.")
        else:
            st.markdown("#### Select Drop")
            selected = st.selectbox("", [drop["name"] for drop in st.session_state.drops], label_visibility="collapsed")
            selected_drop = next(d for d in st.session_state.drops if d["name"] == selected)
            submissions = [s for s in st.session_state.submissions if s["drop_id"] == selected_drop["id"]]

            if not submissions:
                st.info("No submissions yet for this drop.")
            else:
                matched = [s["track_name"] for s in submissions if s["matches"]]
                unmatched = [s["track_name"] for s in submissions if not s["matches"]]

                # Function to render styled song cards
                def render_song_list(song_names, bg_color, text_color="white"):
                    for name in song_names:
                        st.markdown(
                            f"""
                            <div style="background-color:{bg_color};
                                        padding:0.6em 1.2em;
                                        border-radius:8px;
                                        margin-bottom:8px;
                                        font-weight:500;
                                        color:{text_color};
                                        box-shadow:0 2px 5px rgba(0,0,0,0.1);">
                                🎵 {name}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # matched songs list
                st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
                st.markdown("#### Matched Songs")
                if matched:
                    render_song_list(matched, "#f1f3f6", "#333333")
                else:
                    st.markdown("No songs matched the criteria yet.")

                # unmatched songs list
                st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
                st.markdown("#### Unmatched Songs")
                if unmatched:
                    render_song_list(unmatched, "#f1f3f6", "#333333")
                else:
                    st.markdown("All submitted songs met the criteria.")


# producer
elif role == "Producer":
    st.markdown("# Available Drops")

    button_style = """
    <style>
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 0.6em 1.5em;
        border-radius: 10px;
        border: none;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff1c1c;
        transform: scale(1.02);
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    demo_drops = [drop for drop in st.session_state.drops if drop['type'] == 'Demo']
    promo_drops = [drop for drop in st.session_state.drops if drop['type'] == 'Promo']

    def show_drop_cards(drops, label):
        if drops:
            st.markdown(f"### {label}")
            for drop in drops:
                with st.container():
                    st.markdown(
                        f"""
                        <div style="background-color:#f9f9f9;padding:1em;border-radius:10px;margin-bottom:1em;box-shadow:0 2px 5px rgba(0,0,0,0.05);">
                            <div style="font-weight:bold;font-size:1.2em;">{drop['name']}</div>
                            <div style="color:#666;margin-bottom:0.5em;">Genres: {', '.join(drop['genres'])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if st.button("Drop Submission", key=f"select_{drop['id']}"):
                        st.session_state.selected_drop_id = drop["id"]
                        st.rerun()                  

                    if st.session_state.get("selected_drop_id") == drop["id"]:
                        audio_file = st.file_uploader("", type=["wav", "mp3"], key=f"uploader_{drop['id']}", label_visibility="collapsed")
                        if audio_file is not None:
                            processed_path = convert_audio_to_30s_segment(audio_file)
                            audio = MonoLoader(filename=processed_path, sampleRate=16000)()
                            activations, labels = model.predict_labels(audio)
                            score_map = dict(zip(labels, activations))

                            # genre matching
                            genre_scores = [(g, score_map.get(f"Electronic---{g}", 0.0)) for g in drop["genres"]]
                            df = pd.DataFrame(genre_scores, columns=["Genre", "Score"]).sort_values("Score", ascending=False)

                            matches = [g for g, s in genre_scores if s >= drop["thresholds"][g]]
                            if matches:
                                st.success(f"Your track fits the drop! Thank you for your submission.")
                            else:
                                st.error("Your track does not meet this drop's criteria.")
                            
                            # save submissions
                            st.session_state.submissions.append({
                                "drop_id": drop["id"],
                                "drop_name": drop["name"],
                                "track_name": audio_file.name,
                                "scores": score_map,
                                "matches": matches,
                                "non_matches": [g for g in drop["genres"] if g not in matches]
                            })
                                
                        st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        else:
            st.info(f"No {label.lower()} available yet.")

    show_drop_cards(demo_drops, "Demo Drops")
    show_drop_cards(promo_drops, "Promo Drops")