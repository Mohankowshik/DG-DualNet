document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(`tab-${btn.dataset.tab}`);
      if (target) target.classList.add("active");
    });
  });

  // UI Elements
  const uploadZone = document.getElementById("upload-zone");
  const fileInput = document.getElementById("file-input");
  const btnAnalyze = document.getElementById("btn-analyze");
  const thresholdSlider = document.getElementById("threshold-slider");
  const thresholdVal = document.getElementById("threshold-val");
  const sampleButtons = document.querySelectorAll(".btn-sample");

  // Output Elements
  const verdictBanner = document.getElementById("verdict-banner");
  const verdictText = document.getElementById("verdict-text");
  const verdictSubtext = document.getElementById("verdict-subtext");
  const confidencePct = document.getElementById("confidence-pct");

  const imgFace = document.getElementById("img-face");
  const imgGradcam = document.getElementById("img-gradcam");
  const imgFft = document.getElementById("img-fft");
  const imgResidual = document.getElementById("img-residual");
  const imgRadialChart = document.getElementById("img-radial-chart");
  const domainBarsContainer = document.getElementById("domain-attribution-bars");
  const indicatorsList = document.getElementById("indicators-list");

  let currentSelectedFile = null;
  let currentSampleName = "sample_real.png";

  // Threshold slider
  thresholdSlider.addEventListener("input", (e) => {
    thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // File Upload Handlers
  uploadZone.addEventListener("click", () => fileInput.click());

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  });

  function handleFileSelection(file) {
    currentSelectedFile = file;
    currentSampleName = null;
    uploadZone.querySelector("p").textContent = `Selected: ${file.name}`;
    runDetection();
  }

  // Preloaded Samples
  sampleButtons.forEach(btn => {
    if (btn.dataset.sample) {
      btn.addEventListener("click", () => {
        currentSampleName = btn.dataset.sample;
        currentSelectedFile = null;
        uploadZone.querySelector("p").textContent = `Selected Sample: ${currentSampleName}`;
        runDetection();
      });
    }
  });

  btnAnalyze.addEventListener("click", () => {
    runDetection();
  });

  async function runDetection() {
    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin">
        <line x1="12" y1="2" x2="12" y2="6"></line>
        <line x1="12" y1="18" x2="12" y2="22"></line>
        <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
        <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
        <line x1="2" y1="12" x2="6" y2="12"></line>
        <line x1="18" y1="12" x2="22" y2="12"></line>
        <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line>
        <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
      </svg>
      Analyzing Dual-Stream Forensics...
    `;

    try {
      let data;
      const threshold = parseFloat(thresholdSlider.value);

      if (currentSelectedFile) {
        const formData = new FormData();
        formData.append("file", currentSelectedFile);
        formData.append("threshold", threshold);
        const res = await fetch("/api/detect", {
          method: "POST",
          body: formData
        });
        data = await res.json();
      } else if (currentSampleName) {
        const res = await fetch(`/api/sample/${currentSampleName}?threshold=${threshold}`);
        data = await res.json();
      } else {
        alert("Please select a sample or upload an image.");
        return;
      }

      updateUIWithResults(data);
    } catch (err) {
      console.error(err);
      alert("Error performing detection: " + err.message);
    } finally {
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
          <circle cx="12" cy="12" r="10"/>
          <polygon points="10 8 16 12 10 16 10 8"/>
        </svg>
        Run Forensic Diagnostics
      `;
    }
  }

  function updateUIWithResults(data) {
    if (!data || !data.visualizations) return;

    // 1. Verdict Banner
    if (data.is_fake) {
      verdictBanner.className = "verdict-banner fake";
      verdictText.textContent = "DEEPFAKE DETECTED";
      verdictSubtext.textContent = "High anomalous frequency response & spatial manipulation cues present";
    } else {
      verdictBanner.className = "verdict-banner real";
      verdictText.textContent = "AUTHENTIC REAL";
      verdictSubtext.textContent = "Natural organic texture distribution; consistent Fourier radial decay";
    }

    const confPct = (data.confidence * 100).toFixed(1);
    confidencePct.textContent = `${confPct}%`;

    // 2. Visualizations
    imgFace.src = data.visualizations.face_crop;
    imgGradcam.src = data.visualizations.gradcam_overlay;
    imgFft.src = data.visualizations.fft_spectrum;
    imgResidual.src = data.visualizations.high_pass_residual;
    imgRadialChart.src = data.visualizations.radial_chart;

    // 3. Domain Attribution Bars
    domainBarsContainer.innerHTML = "";
    if (data.domain_breakdown) {
      Object.entries(data.domain_breakdown).forEach(([name, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <div class="bar-label-group">
            <span style="color: var(--text-muted);">${formatDomainName(name)}</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--text-main);">${pct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width: ${pct}%;"></div>
          </div>
        `;
        domainBarsContainer.appendChild(row);
      });
    }

    // 4. Indicators
    indicatorsList.innerHTML = "";
    if (data.indicators && data.indicators.length > 0) {
      data.indicators.forEach(ind => {
        const li = document.createElement("li");
        li.textContent = ind;
        indicatorsList.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.textContent = "No significant synthetic anomalies detected.";
      indicatorsList.appendChild(li);
    }
  }

  function formatDomainName(raw) {
    return raw.replace("_", " ");
  }

  // Load first default sample
  setTimeout(() => {
    runDetection();
  }, 400);
});
