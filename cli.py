"""
Command Line Interface (CLI) for TruthPulse Fake News Detector.
Usage:
  python cli.py --title "Headline here" --text "Article body here"
  python cli.py --sample fake
  python cli.py --sample real
  python cli.py --interactive
"""

import sys
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import FakeNewsModelManager
from preprocessing import extract_linguistic_features
from explainability import explain_prediction

def run_cli():
    parser = argparse.ArgumentParser(description="TruthPulse Fake News Detector CLI")
    parser.add_argument("--title", type=str, default="", help="Article title / headline")
    parser.add_argument("--text", type=str, default="", help="Article body content")
    parser.add_argument("--sample", type=str, choices=["real", "fake"], help="Test with built-in sample article")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal prompt")

    args = parser.parse_args()

    manager = FakeNewsModelManager()
    manager.load_artifacts()

    if args.sample == "real":
        title = "FDA Approves Breakthrough Gene Therapy for Rare Sickle Cell Disease"
        text = "The Food and Drug Administration has granted regulatory approval to a groundbreaking CRISPR-based gene therapy designed to treat patients aged 12 and older suffering from severe sickle cell disease. Clinical trials demonstrated high safety and sustained efficacy."
    elif args.sample == "fake":
        title = "SHOCKING SECRET: Common Kitchen Lemon Juice Completely Cures All Cancers in 48 Hours!"
        text = "Big Pharma doesn't want you to know this suppressed natural secret! Top whistleblowers inside classified medical institutions have revealed that drinking hot boiled lemon water infused with baking soda destroys 100% of cancer cells within 48 hours!"
    elif args.interactive:
        print("\n==================================================")
        print("📰 TruthPulse Fake News Detector - Interactive CLI")
        print("==================================================")
        title = input("\nEnter Article Headline: ").strip()
        print("\nEnter Article Body Text (Press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        text = "\n".join(lines).strip()
    else:
        title = args.title
        text = args.text

    if not title and not text:
        parser.print_help()
        print("\n[!] Example: python cli.py --sample fake")
        return

    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Headline: {title[:70]}...")
    print(f"Content length: {len(text)} characters\n")

    pred = manager.predict(title, text)
    ling = extract_linguistic_features(text, title)
    words = explain_prediction(f"{title} {text}", manager.best_model, manager.vectorizer, top_k=6)

    status_icon = "[!]" if pred["label"] == 1 else "[+]"
    print(f"Verdict:           {status_icon} {pred['verdict'].upper()}")
    print(f"Confidence:        {pred['confidence']}%")
    print(f"Model:             {pred['model_used']}")
    print(f"Sensationalism:    {ling['sensationalism_score']}/100")
    print(f"All-Caps Ratio:    {ling['uppercase_ratio']}%")
    print(f"Exclamations:      {ling['exclamation_density']}%")

    if words:
        print("\nTop Contributing Keyword Signals:")
        for w in words:
            tag = "[FAKE SIGNAL]" if w["indicates"].startswith("Fake") else "[REAL SIGNAL]"
            print(f" - {w['word']:15s} {tag} (weight: {w['impact']})")

    print("="*50 + "\n")

if __name__ == "__main__":
    run_cli()
