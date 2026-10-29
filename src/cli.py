"""
Unified Command-Line Interface (CLI) for Domain-Generalized Deepfake Detection.
Commands:
  detect   - Inspect an image or folder and output forensic diagnostics
  train    - Train DG-DualNet across source domains with GRL adversarial alignment
  evaluate - Run Leave-One-Domain-Out cross-generator benchmark
  web      - Launch the interactive forensic dashboard web application
"""

import os
import sys
import argparse
import json
import cv2
import torch

from .inference.detector import DeepfakeDetector
from .models.dg_dualnet import DGDualNet
from .data.dataset import build_domain_dataloaders
from .data.synthetic_domains import DOMAIN_NAMES
from .training.trainer import DGTrainer
from .training.evaluate import evaluate_model


def run_detect(args):
    print(f"\n[+] Loading DG-DualNet Detector (weights: {args.weights})...")
    detector = DeepfakeDetector(weights_path=args.weights, backbone=args.backbone, device=args.device)

    print(f"[+] Analyzing image: {args.image} (threshold: {args.threshold})...")
    result = detector.analyze_image(args.image, threshold=args.threshold)

    print("\n" + "=" * 50)
    print(f"  FORENSIC VERDICT:    {result['verdict']}")
    print(f"  CONFIDENCE:          {result['confidence'] * 100:.2f}%")
    print(f"  FAKE PROBABILITY:    {result['prob_fake'] * 100:.2f}%")
    print(f"  REAL PROBABILITY:    {result['prob_real'] * 100:.2f}%")
    print(f"  FACE DETECTED:       {result['face_detected']} (BBox: {result['bbox']})")
    print("=" * 50)

    print("\n[+] Generator Attribution Likelihoods:")
    for dom, prob in result["domain_breakdown"].items():
        print(f"  - {dom:<22}: {prob * 100:.2f}%")

    if result["indicators"]:
        print("\n[+] Diagnostic Indicators:")
        for ind in result["indicators"]:
            print(f"  * {ind}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save face crop and heatmap
        import base64
        for name, data_uri in result["visualizations"].items():
            if "," in data_uri:
                b64 = data_uri.split(",", 1)[1]
                save_file = os.path.join(args.output_dir, f"{name}.png")
                with open(save_file, "wb") as f:
                    f.write(base64.b64decode(b64))
        print(f"\n[+] Saved forensic visual maps to '{args.output_dir}'")


def run_train(args):
    print(f"\n[+] Initializing Domain-Generalized Training Pipeline...")
    train_loader, val_loader, test_ood_loader = build_domain_dataloaders(
        data_dir=args.data,
        leave_out_domain=args.leave_out,
        batch_size=args.batch_size
    )

    model = DGDualNet(
        backbone_name=args.backbone,
        pretrained_spatial=args.pretrained,
        feature_dim=args.feature_dim,
        num_domains=len(DOMAIN_NAMES),
        fusion_type=args.fusion
    )

    trainer = DGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_ood_loader=test_ood_loader,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        max_dann_lambda=args.max_lambda
    )

    results = trainer.fit(epochs=args.epochs, save_best_name=args.save_name)
    print(f"\n[+] Training completed! Best checkpoint: {results['best_checkpoint']}")
    print(f"[+] Best Validation AUC: {results['best_val_auc']:.4f}")


def run_evaluate(args):
    print(f"\n[+] Running Cross-Generator Evaluation...")
    _, val_loader, test_ood_loader = build_domain_dataloaders(
        data_dir=args.data,
        leave_out_domain=args.leave_out,
        batch_size=args.batch_size
    )

    model = DGDualNet(backbone_name=args.backbone, num_domains=len(DOMAIN_NAMES))
    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location=args.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"[+] Loaded checkpoint weights from {args.weights}")
    dev = torch.device(args.device)
    model.to(dev)

    print("\n--- In-Domain Validation Metrics ---")
    val_metrics = evaluate_model(model, val_loader, dev)
    print(f"  Accuracy: {val_metrics['accuracy'] * 100:.2f}% | AUC: {val_metrics['auc']:.4f} | EER: {val_metrics['eer'] * 100:.2f}%")

    print("\n--- Out-of-Domain (Unseen Generator) Metrics ---")
    ood_metrics = evaluate_model(model, test_ood_loader, dev)
    print(f"  Accuracy: {ood_metrics['accuracy'] * 100:.2f}% | AUC: {ood_metrics['auc']:.4f} | EER: {ood_metrics['eer'] * 100:.2f}%")
    print(f"  Confusion Matrix: {ood_metrics['confusion_matrix']}")


def run_web(args):
    import uvicorn
    print(f"\n[+] Launching DG-DualNet Forensic Web Server at http://{args.host}:{args.port}")
    uvicorn.run("src.web.app:app", host=args.host, port=args.port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(description="DG-DualNet Domain-Generalized Deepfake Detection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Detect
    p_detect = subparsers.add_parser("detect", help="Run forensic detection on an image")
    p_detect.add_argument("--image", type=str, required=True, help="Path to input face image")
    p_detect.add_argument("--weights", type=str, default="weights/dg_dualnet_best.pt", help="Path to model weights")
    p_detect.add_argument("--backbone", type=str, default="resnet18", help="Spatial backbone")
    p_detect.add_argument("--threshold", type=float, default=0.50, help="Classification threshold")
    p_detect.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    p_detect.add_argument("--output-dir", type=str, default="output", help="Directory to save visualizer maps")

    # Train
    p_train = subparsers.add_parser("train", help="Train DG-DualNet with Domain Adversarial Alignment")
    p_train.add_argument("--data", type=str, default="data/synthetic", help="Data directory")
    p_train.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p_train.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p_train.add_argument("--leave-out", type=int, default=3, help="Domain ID to leave out as unseen target")
    p_train.add_argument("--backbone", type=str, default="resnet18")
    p_train.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    p_train.add_argument("--feature-dim", type=int, default=512)
    p_train.add_argument("--fusion", type=str, default="cross_attention", choices=["cross_attention", "gated"])
    p_train.add_argument("--max-lambda", type=float, default=1.0, help="Max GRL lambda")
    p_train.add_argument("--save-dir", type=str, default="weights")
    p_train.add_argument("--save-name", type=str, default="dg_dualnet_best.pt")
    p_train.add_argument("--device", type=str, default="cpu")

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model generalization")
    p_eval.add_argument("--data", type=str, default="data/synthetic")
    p_eval.add_argument("--weights", type=str, default="weights/dg_dualnet_best.pt")
    p_eval.add_argument("--leave-out", type=int, default=3)
    p_eval.add_argument("--batch-size", type=int, default=8)
    p_eval.add_argument("--backbone", type=str, default="resnet18")
    p_eval.add_argument("--device", type=str, default="cpu")

    # Web
    p_web = subparsers.add_parser("web", help="Start the interactive web dashboard")
    p_web.add_argument("--host", type=str, default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument("--reload", action="store_true")

    args = parser.parse_args()
    if args.command == "detect":
        run_detect(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "web":
        run_web(args)


if __name__ == "__main__":
    main()
