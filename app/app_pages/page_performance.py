"""Page 4 – Model Performance Dashboard"""
import streamlit as st
import numpy as np
from pathlib import Path
from .utils import MODELS, MODEL_COLORS, load_classes, load_single_model

def show():
    st.markdown("<div class='hero-title' style='font-size:2.2rem;'>📊 Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle' style='font-size:1rem;'>Compare accuracy, speed, and architecture across all CNN models</div>", unsafe_allow_html=True)

    class_names = load_classes()

    # ── Architecture Overview ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🏗️ Architecture Overview</div>", unsafe_allow_html=True)
    arch_data = {
        "MobileNetV2":    {"params": "3.4M", "depth": 53,  "imagenet_acc": "71.8%", "speed": "⚡⚡⚡⚡", "size_mb": "14", "color": MODEL_COLORS["MobileNetV2"]},
        "VGG16":          {"params": "138M", "depth": 23,  "imagenet_acc": "71.3%", "speed": "⚡⚡",   "size_mb": "528","color": MODEL_COLORS["VGG16"]},
        "ResNet50":       {"params": "25.6M","depth": 107, "imagenet_acc": "74.9%", "speed": "⚡⚡⚡",  "size_mb": "98", "color": MODEL_COLORS["ResNet50"]},
        "EfficientNetB0": {"params": "5.3M", "depth": 132, "imagenet_acc": "77.1%", "speed": "⚡⚡⚡",  "size_mb": "29", "color": MODEL_COLORS["EfficientNetB0"]},
    }

    cols = st.columns(4)
    for col, (name, info) in zip(cols, arch_data.items()):
        with col:
            color = info["color"]
            # Check if model file exists
            model_path = MODELS[name]
            exists = model_path.exists()
            status_badge = f"<span style='color:#00ff88; font-size:0.7rem;'>✅ Trained</span>" if exists else \
                           f"<span style='color:#ff6b6b; font-size:0.7rem;'>❌ Not found</span>"
            st.markdown(f"""
            <div class='feature-card' style='border-color:{color}44;'>
                <div style='color:{color}; font-weight:700; font-size:1rem;'>{name}</div>
                {status_badge}
                <hr style='border-color:#2d3748; margin:0.8rem 0;'/>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; font-size:0.82rem;'>
                    <div>
                        <div style='color:#8b949e;'>Parameters</div>
                        <div style='color:#e6edf3; font-weight:600;'>{info["params"]}</div>
                    </div>
                    <div>
                        <div style='color:#8b949e;'>Depth</div>
                        <div style='color:#e6edf3; font-weight:600;'>{info["depth"]} layers</div>
                    </div>
                    <div>
                        <div style='color:#8b949e;'>ImageNet Acc.</div>
                        <div style='color:{color}; font-weight:600;'>{info["imagenet_acc"]}</div>
                    </div>
                    <div>
                        <div style='color:#8b949e;'>Speed</div>
                        <div style='color:#e6edf3;'>{info["speed"]}</div>
                    </div>
                    <div>
                        <div style='color:#8b949e;'>File Size</div>
                        <div style='color:#e6edf3; font-weight:600;'>{info["size_mb"]} MB</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── File sizes of actual trained models ───────────────────────────────────
    st.markdown("<div class='section-header'>💾 Trained Model Files</div>", unsafe_allow_html=True)
    import pandas as pd
    model_files = []
    for name, path in MODELS.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            model_files.append({
                "Model": name,
                "File": path.name,
                "Size (MB)": f"{size_mb:.1f}",
                "Status": "✅ Available",
            })
        else:
            model_files.append({
                "Model": name,
                "File": path.name,
                "Size (MB)": "—",
                "Status": "❌ Not found",
            })
    st.dataframe(pd.DataFrame(model_files), use_container_width=True, hide_index=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Inference Speed Benchmark (live) ──────────────────────────────────────
    st.markdown("<div class='section-header'>⚡ Inference Speed Benchmark</div>", unsafe_allow_html=True)
    st.info("Click the button below to run a live inference speed benchmark using a random test image.")

    if st.button("🚀 Run Speed Benchmark", use_container_width=True):
        import time
        test_img = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8).astype("float32")
        speed_results = []

        prog = st.progress(0, text="Benchmarking…")
        for i, (name, path) in enumerate(MODELS.items()):
            prog.progress(i / len(MODELS), text=f"Benchmarking {name}…")
            if not path.exists():
                speed_results.append({"Model": name, "Avg. Time (ms)": "Model not found", "FPS": "—"})
                continue
            model = load_single_model(str(path))
            if model is None:
                speed_results.append({"Model": name, "Avg. Time (ms)": "Load failed", "FPS": "—"})
                continue

            # Warm up
            _ = model.predict(test_img, verbose=0)

            # Benchmark
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                model.predict(test_img, verbose=0)
                times.append((time.perf_counter() - t0) * 1000)
            avg_ms = np.mean(times)
            fps = 1000 / avg_ms
            speed_results.append({"Model": name, "Avg. Time (ms)": f"{avg_ms:.1f}", "FPS": f"{fps:.1f}"})
        prog.empty()

        df_speed = pd.DataFrame(speed_results)
        st.dataframe(df_speed, use_container_width=True, hide_index=True)

        # Bar chart
        st.markdown("#### ⏱️ Inference Time Comparison")
        valid = [(r["Model"], float(r["Avg. Time (ms)"])) for r in speed_results if r["Avg. Time (ms)"] not in ("Model not found", "Load failed")]
        if valid:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("#0a0e1a")
            ax.set_facecolor("#0d1117")
            names = [v[0] for v in valid]
            times_ = [v[1] for v in valid]
            colors = [MODEL_COLORS.get(n, "#00d4ff") for n in names]
            bars = ax.barh(names, times_, color=colors, height=0.5)
            ax.set_xlabel("Milliseconds (ms)", color="#8b949e")
            ax.tick_params(colors="#8b949e")
            ax.spines[:].set_color("#2d3748")
            for bar, t in zip(bars, times_):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{t:.1f}ms", va="center", color="#e6edf3", fontsize=9)
            st.pyplot(fig)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Class Distribution ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📦 Dataset Class Distribution</div>", unsafe_allow_html=True)
    from .utils import DATA_DIR
    train_dir = DATA_DIR / "classification" / "train"
    if train_dir.exists():
        class_counts = {}
        for cls_dir in sorted(train_dir.iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob("*.*")))
                class_counts[cls_dir.name] = count

        if class_counts:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor("#0a0e1a")
            ax.set_facecolor("#0d1117")
            names_sorted = list(class_counts.keys())
            counts = [class_counts[n] for n in names_sorted]
            # Color gradient
            colors = [f"#{int(0 + i*(139/len(names_sorted))):02x}{int(212 - i*(100/len(names_sorted))):02x}ff" for i in range(len(names_sorted))]
            ax.bar(names_sorted, counts, color="#00d4ff", edgecolor="#1e2d45", linewidth=0.5)
            ax.set_xlabel("Class", color="#8b949e")
            ax.set_ylabel("Image Count", color="#8b949e")
            ax.tick_params(axis="x", rotation=45, labelcolor="#8b949e", labelsize=7.5)
            ax.tick_params(axis="y", labelcolor="#8b949e")
            ax.spines[:].set_color("#2d3748")
            ax.grid(axis="y", color="#1e2d45", linestyle="--", alpha=0.5)
            st.pyplot(fig)

            total = sum(counts)
            st.markdown(f"**Total training images:** `{total:,}` across `{len(class_counts)}` classes")
    else:
        st.info("📂 Training data not found for class distribution analysis.")
