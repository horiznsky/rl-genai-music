import argparse
import yaml
import numpy as np
from src.generator_pipeline import MusicGenPipeline

def main():
    parser = argparse.ArgumentParser(description="Generate counter-melodies via trained Policy Models.")
    parser.add_argument("--model", type=str, default="actor_critic_model_final.pth")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--vocab", type=str, default="pitch_vocab_single.json")
    parser.add_argument("--output", type=str, default="output.mid")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("Initializing production generation pipeline...")
    pipeline = MusicGenPipeline(args.model, config, args.vocab)
    
    print("Synthesizing baseline input melody track...")
    scale = [60, 62, 64, 65, 67, 69, 71, 72]  
    synthetic_human_melody = list(np.random.choice(scale, size=64, replace=True))
    
    print("Executing policy model generation cycle...")
    generated_stream = pipeline.generate_counter_melody(synthetic_human_melody, config["data"]["ticks_per_quarter"])
    
    generated_stream.write('midi', fp=args.output)
    print(f"🎉 Success! Generated MIDI output saved elegantly to: {args.output}")

if __name__ == "__main__":
    main()