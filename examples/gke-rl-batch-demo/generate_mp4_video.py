#!/usr/bin/env python3
"""
GKE CATS 5-Part Deep-Dive — MP4 Video Generator with High-Clarity Spoken Audio
================================================================================
Generates standalone MP4 video files complete with crystal-clear synthesized spoken English
voiceover narration using FFmpeg's built-in 'flite' text-to-speech engine (slt voice),
resampled to 48kHz stereo AAC at 192 kbps to eliminate aliasing and concatenation garbling.
"""

import os
import sys
import argparse
import subprocess
import tempfile

def create_video(output_path: str, duration_mode: str = "fast"):
    if duration_mode == "full":
        durations = [35, 35, 40, 35, 35]
        fps = 24
    else:
        durations = [9, 9, 10, 9, 8]  # Total 45s (Executive Summary with Audio)
        fps = 24

    sections = [
        {
            "title": "SECTION 1: THE PROBLEM",
            "sub": "RL Trainers Leave GPUs Idle 85.7% of the Time",
            "detail": "Only 14.3% GPU Duty Cycle (20s training / 140s cycle)",
            "voice": "Section 1. The problem with reinforcement learning trainer idle cycles. Traditional R L jobs leave expensive GPUs idle 85 percent of the time during evaluation and rollout generation.",
            "color": "0x0f172a",
            "accent": "0xef4444"
        },
        {
            "title": "SECTION 2: OPPORTUNITIES",
            "sub": "Harvest Idle GPU Valleys Without Custom Kernels",
            "detail": "Boost Node Utilization to 98.6% with Zero Priority Inversion",
            "voice": "Section 2. Opportunities opened by time slicing. Cooperative time slicing converts idle gaps into productive batch inference, elevating node duty cycle from 14 percent to 98.6 percent.",
            "color": "0x064e3b",
            "accent": "0x10b981"
        },
        {
            "title": "SECTION 3: SETUP & CODE",
            "sub": "Kubernetes DRA ResourceClaim (ExactCount=1)",
            "detail": "Shared L4 Node Pool -> rl-trainer & shadow-vllm Pods",
            "voice": "Section 3. Setup and cluster mapping. A single Dynamic Resource Allocation Resource Claim shares one NVIDIA L4 GPU between the R L Trainer and Shadow v L L M pods.",
            "color": "0x1e1b4b",
            "accent": "0x3b82f6"
        },
        {
            "title": "SECTION 4: DUTY CYCLE",
            "sub": "14.3% Standalone RL vs 98.6% Interleaved CATS",
            "detail": "Stock vLLM Batch Inference Runs During 120s RL Idle Gap",
            "voice": "Section 4. Duty cycle comparison. Standalone R L yields only 14.3 percent duty cycle. With CATS interleaving, effective node utilization reaches 98.6 percent.",
            "color": "0x451a03",
            "accent": "0xf59e0b"
        },
        {
            "title": "SECTION 5: TIMELINE HANDSHAKE",
            "sub": "Second-by-Second Preemption Sequence",
            "detail": "acquire() -> SIGTERM yield (<200ms) -> Training -> Auto Resume",
            "voice": "Section 5. Detailed timeline handshake. The instant R L Trainer requests access, v L L M cleanly yields memory within 200 milliseconds, ensuring zero priority inversion.",
            "color": "0x1f2937",
            "accent": "0x06b6d4"
        }
    ]

    print(f"🎬 Generating '{output_path}' with high-clarity 48kHz stereo audio ({duration_mode.upper()} mode)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_files = []
        for idx, (s, dur) in enumerate(zip(sections, durations)):
            clip_path = os.path.join(tmpdir, f"section_{idx}.mp4")
            clip_files.append(clip_path)

            def esc(txt):
                return txt.replace("'", "").replace(":", "\\:").replace("[", "\\[").replace("]", "\\]")

            t_title = esc(s["title"])
            t_sub = esc(s["sub"])
            t_detail = esc(s["detail"])
            v_text = esc(s["voice"])

            vf = (
                f"drawtext=text='{t_title}':fontcolor={s['accent']}:fontsize=52:x=(w-text_w)/2:y=180,"
                f"drawtext=text='{t_sub}':fontcolor=white:fontsize=32:x=(w-text_w)/2:y=310,"
                f"drawtext=text='{t_detail}':fontcolor=0xcccccc:fontsize=24:x=(w-text_w)/2:y=440,"
                f"drawtext=text='GKE Cooperative Acceleration Time-Slicing (CATS)':fontcolor=0x64748b:fontsize=20:x=40:y=h-50,"
                f"drawtext=text='Time\\: %{{pts\\:hms}}':fontcolor=0x64748b:fontsize=20:x=w-180:y=h-50"
            )

            # Audio filter: resample 16kHz flite voice (slt) to 48kHz stereo with boosted gain and silence padding
            af = (
                f"aresample=48000,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                f"volume=2.0,apad=pad_dur={dur}"
            )

            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c={s['color']}:s=1280x720:d={dur}",
                "-f", "lavfi", "-i", f"flite=text='{v_text}':voice=slt",
                "-vf", vf, "-af", af,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-t", str(dur),
                clip_path
            ]

            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res.returncode != 0:
                print(f"❌ Error generating clip {idx}: {res.stderr.decode('utf-8', errors='ignore')}", file=sys.stderr)
                sys.exit(1)
            print(f"  ✔ Rendered Section {idx+1} (48kHz Stereo AAC, {dur}s)")

        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w") as f:
            for c in clip_files:
                f.write(f"file '{c}'\n")

        # Concatenate and re-encode audio to 48kHz stereo AAC to prevent packet boundary garbling
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            output_path
        ]
        res = subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"❌ Error concatenating clips: {res.stderr.decode('utf-8', errors='ignore')}", file=sys.stderr)
            sys.exit(1)

    print(f"\n🎉 Success! Crystal clear video saved to: {os.path.abspath(output_path)}")
    print(f"📊 File size: {os.path.getsize(output_path) / 1024:.1f} KB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 5-Part GKE CATS Demo Video with Crystal Clear Audio")
    parser.add_argument("--out", default="gke_rl_batch_demo_audio.mp4", help="Output MP4 file path")
    parser.add_argument("--full", action="store_true", help="Generate full 3-minute (180s) video")
    parser.add_argument("--fast", action="store_true", help="Generate 45s executive summary video (default)")
    args = parser.parse_args()

    mode = "full" if args.full else "fast"
    create_video(args.out, duration_mode=mode)
