import gradio as gr
import yaml
import tempfile
import music21
import base64
from src.generator_pipeline import MusicGenPipeline

# 1. Initialize the pipeline globally so it loads into memory once
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    pipeline = MusicGenPipeline(
        model_path="actor_critic_model_final.pth", 
        config=config, 
        vocab_path="data/pitch_vocab_single.json"
    )
    model_loaded = True
    print("✅ Pipeline loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Model or config not found. {e}")
    model_loaded = False

# Helper: Convert text notes to MIDI integers
def parse_text_melody(text_melody):
    pitches = []
    tokens = text_melody.replace(',', ' ').split()
    for token in tokens:
        try:
            n = music21.note.Note(token)
            pitches.append(n.pitch.midi)
        except Exception as e:
            print(f"Skipping invalid note '{token}': {e}")
    return pitches

# 2. Define the core generation function
def generate_duet(input_method, user_midi_file, text_melody):
    if not model_loaded:
        return None, "<p style='color:red;'>Model failed to load. Check your weights and config paths.</p>"
        
    human_pitches = []
    
    # METHOD A: Parse uploaded MIDI
    if input_method == "Upload MIDI File" and user_midi_file is not None:
        try:
            score = music21.converter.parse(user_midi_file.name)
            for n in score.flatten().notes:
                if n.isNote:
                    human_pitches.append(n.pitch.midi)
                elif n.isChord:
                    human_pitches.append(n.pitches[-1].midi) # Take the highest note
        except Exception as e:
            return None, f"<p style='color:red;'>Error parsing MIDI: {e}</p>"

    # METHOD B: Parse Text Input
    elif input_method == "Type Melody (Text)" and text_melody.strip():
        human_pitches = parse_text_melody(text_melody)
        if not human_pitches:
            return None, "<p style='color:red;'>No valid notes recognized. Please use format like 'C4 D4 E4'.</p>"

    # Safety check
    if len(human_pitches) == 0:
        return None, "<p style='color:red;'>Please provide a valid MIDI file or type a melody.</p>"
    
    # Run the generator
    generated_stream = pipeline.generate_counter_melody(
        human_pitches, 
        config["data"]["ticks_per_quarter"]
    )
    
    # Save the output to a temporary file
    temp_midi = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
    generated_stream.write('midi', fp=temp_midi.name)

    # Convert the MIDI file to a base64 Data URI for HTML playback
    with open(temp_midi.name, "rb") as f:
        b64_midi = base64.b64encode(f.read()).decode("utf-8")
    midi_data_uri = f"data:audio/midi;base64,{b64_midi}"

    # Inject Google Magenta's HTML MIDI Player
    html_player = f"""
    <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.4.0"></script>
    <div style="padding: 10px; background: #f0f0f0; border-radius: 8px; margin-bottom: 10px;">
        <midi-player src="{midi_data_uri}" sound-font visualizer="#myVisualizer"></midi-player>
    </div>
    <midi-visualizer type="piano-roll" id="myVisualizer" src="{midi_data_uri}" style="width: 100%; height: 250px; background: white; border-radius: 8px; overflow: hidden;"></midi-visualizer>
    """
    
    return temp_midi.name, html_player

# 3. Build the Frontend Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎹 AI Contrapuntal Music Generator
        *An Actor-Critic Reinforcement Learning system that acts as your AI duet partner.*
        
        Provide a baseline melody below, and the multi-objective reward ensemble will generate a harmonically and rhythmically aligned counter-melody to accompany it.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Input & Controls")
            
            input_method = gr.Radio(
                choices=["Upload MIDI File", "Type Melody (Text)"], 
                value="Upload MIDI File", 
                label="Choose Input Method"
            )
            
            user_midi = gr.File(
                label="Upload Custom MIDI Melody", 
                file_types=[".mid", ".midi"],
                visible=True
            )
            
            text_melody = gr.Textbox(
                label="Type Notes (e.g., C4 D4 E4 F4 G4)", 
                placeholder="C4 E4 G4 C5...",
                lines=2,
                visible=False
            )
            
            # Dynamic UI update based on Radio selection
            def update_ui(method):
                if method == "Upload MIDI File":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)
                    
            input_method.change(fn=update_ui, inputs=input_method, outputs=[user_midi, text_melody])
            
            generate_btn = gr.Button("Generate AI Duet", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 🎵 Output & Playback")
            
            html_output = gr.HTML(label="Piano Roll Visualizer")
            output_midi = gr.File(label="Download Generated AI Duet", interactive=False)
            
    # Link the button to the generation function
    generate_btn.click(
        fn=generate_duet, 
        inputs=[input_method, user_midi, text_melody], 
        outputs=[output_midi, html_output]
    )

if __name__ == "__main__":
    # share=True gives you a public link to share on your resume/portfolio!
    demo.launch(share=False)