async function uploadForm(url, form) {
  const formData = new FormData(form);
  const renderToggle = form.querySelector('input[name="render_artifacts"]');
  if (renderToggle) {
    formData.set("render_artifacts", renderToggle.checked ? "true" : "false");
  }
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed");
  }
  return payload;
}

function formatValue(value, digits = 3) {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "n/a";
    }
    if (Number.isInteger(value)) {
      return String(value);
    }
    return value.toFixed(digits);
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  return String(value);
}

function buildMetric(label, value, hint = "") {
  const card = document.createElement("div");
  card.className = "metric-card";

  const title = document.createElement("p");
  title.className = "metric-label";
  title.textContent = label;

  const body = document.createElement("p");
  body.className = "metric-value";
  body.textContent = value;

  card.append(title, body);

  if (hint) {
    const foot = document.createElement("p");
    foot.className = "metric-hint";
    foot.textContent = hint;
    card.appendChild(foot);
  }

  return card;
}

function buildStatusChip(text, tone = "neutral") {
  const chip = document.createElement("span");
  chip.className = `status-chip status-${tone}`;
  chip.textContent = text;
  return chip;
}

function clearNode(node) {
  node.innerHTML = "";
}

function renderImages(targetId, items) {
  const container = document.getElementById(targetId);
  container.innerHTML = "";
  items
    .filter((item) => item && item.url)
    .forEach((item) => {
      const card = document.createElement("div");
      card.className = "gallery-item";

      const label = document.createElement("p");
      label.className = "gallery-label";
      label.textContent = item.label;

      const img = document.createElement("img");
      img.src = item.url;
      img.loading = "lazy";

      card.append(label, img);
      container.appendChild(card);
    });
}

function renderState(targetId, title, description, tone = "neutral") {
  const container = document.getElementById(targetId);
  clearNode(container);

  const panel = document.createElement("div");
  panel.className = "result-card";

  const header = document.createElement("div");
  header.className = "result-header";
  header.append(
    buildStatusChip(title, tone),
  );

  const body = document.createElement("p");
  body.className = "result-copy";
  body.textContent = description;

  panel.append(header, body);
  container.appendChild(panel);
}

function renderError(targetId, errorMessage) {
  renderState(targetId, "Request failed", errorMessage, "error");
}

function renderRawJson(payload) {
  const details = document.createElement("details");
  details.className = "raw-json";

  const summary = document.createElement("summary");
  summary.textContent = "Raw JSON";

  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(payload, null, 2);

  details.append(summary, pre);
  return details;
}

function render2dResult(targetId, payload) {
  const container = document.getElementById(targetId);
  clearNode(container);

  const panel = document.createElement("div");
  panel.className = "result-card";

  const header = document.createElement("div");
  header.className = "result-header";
  const tone = payload.status === "completed" ? "success" : "neutral";
  header.append(
    buildStatusChip(payload.status || "unknown", tone),
    buildStatusChip(payload.processing_node || "2D node", "neutral"),
  );

  const copy = document.createElement("p");
  copy.className = "result-copy";
  copy.textContent = payload.render_artifacts
    ? "Artifacts were saved for visual review."
    : "Fast mode enabled: overlays and masks were not saved.";

  const metrics = document.createElement("div");
  metrics.className = "metric-grid";
  metrics.append(
    buildMetric("Job ID", payload.job_id || "n/a"),
    buildMetric("Inference, s", formatValue(payload.detection?.inference_seconds)),
    buildMetric("Pixel spacing, mm", formatValue(payload.pixel_spacing_mm, 2)),
    buildMetric("Drift level", payload.drift?.level || "n/a"),
    buildMetric("Drift score", formatValue(payload.drift?.combined_score)),
    buildMetric("Mask area, mm2", formatValue(payload.segmentation?.mask_area_mm2, 2)),
    buildMetric("Min diameter, mm", formatValue(payload.segmentation?.min_diameter_mm, 2)),
    buildMetric("Mean diameter, mm", formatValue(payload.segmentation?.mean_diameter_mm, 2)),
    buildMetric("Max diameter, mm", formatValue(payload.segmentation?.max_diameter_mm, 2)),
    buildMetric("Ratio min / mean, %", formatValue(payload.segmentation?.stenosis_ratio_percent, 2)),
    buildMetric("Narrowing, %", formatValue(payload.segmentation?.stenosis_narrowing_percent, 2)),
  );

  panel.append(header, copy, metrics, renderRawJson(payload));
  container.appendChild(panel);
}

function render3dResult(targetId, payload) {
  const container = document.getElementById(targetId);
  clearNode(container);

  const panel = document.createElement("div");
  panel.className = "result-card";

  const tone =
    payload.status === "completed"
      ? "success"
      : payload.status === "failed"
        ? "error"
        : "neutral";

  const header = document.createElement("div");
  header.className = "result-header";
  header.append(
    buildStatusChip(payload.status || "unknown", tone),
    buildStatusChip(payload.job_id || "3D job", "neutral"),
  );

  const copy = document.createElement("p");
  copy.className = "result-copy";
  copy.textContent =
    payload.status === "completed"
      ? "3D job finished and artifacts are ready."
      : "The 3D task is queued or running in the background worker.";

  const metrics = document.createElement("div");
  metrics.className = "metric-grid";
  metrics.append(
    buildMetric("Pixel spacing, mm", formatValue(payload.pixel_spacing_mm, 2)),
    buildMetric("Runtime, s", formatValue(payload.inference_seconds)),
    buildMetric("Min diameter, mm", formatValue(payload.geometry?.min_diameter_mm, 2)),
    buildMetric("Max diameter, mm", formatValue(payload.geometry?.max_diameter_mm, 2)),
    buildMetric("Mean diameter, mm", formatValue(payload.geometry?.mean_diameter_mm, 2)),
    buildMetric("Ratio min / mean, %", formatValue(payload.geometry?.stenosis_ratio_percent, 2)),
    buildMetric("Narrowing, %", formatValue(payload.geometry?.stenosis_narrowing_percent, 2)),
  );

  panel.append(header, copy, metrics, renderRawJson(payload));
  container.appendChild(panel);
}

async function pollJob(jobId) {
  while (true) {
    const response = await fetch(`/api/v1/jobs/${jobId}`);
    const payload = await response.json();
    if (payload.status === "completed" || payload.status === "failed") {
      return payload;
    }
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

function setBusy(form, busy, text) {
  const button = form.querySelector('button[type="submit"]');
  if (!button) {
    return;
  }
  if (!button.dataset.idleLabel) {
    button.dataset.idleLabel = button.textContent;
  }
  button.disabled = busy;
  button.textContent = busy ? text : button.dataset.idleLabel;
}

document.getElementById("form-2d").addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(event.target, true, "Running 2D analysis...");
  renderState("result-2d", "Running", "The request is being processed by one of the 2D nodes.");
  renderImages("gallery-2d", []);
  try {
    const payload = await uploadForm("/api/v1/analyze/2d", event.target);
    render2dResult("result-2d", payload);
    renderImages("gallery-2d", [
      { label: "Original image", url: payload.input?.image_url },
      { label: "Detection overlay", url: payload.artifacts?.detection_overlay_url },
      { label: "Binary vessel mask", url: payload.segmentation?.mask_url },
      { label: "Rendered vessel mask", url: payload.segmentation?.mask_render_url },
      { label: "Segmentation overlay", url: payload.segmentation?.overlay_url },
    ]);
  } catch (error) {
    renderError("result-2d", error.message);
  } finally {
    setBusy(event.target, false);
  }
});

document.getElementById("form-3d").addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(event.target, true, "Queueing 3D analysis...");
  renderState("result-3d", "Queued", "The 3D task was sent to Kafka and is waiting for the worker.");
  renderImages("gallery-3d", []);
  try {
    const queued = await uploadForm("/api/v1/analyze/3d", event.target);
    render3dResult("result-3d", queued);
    const finalJob = await pollJob(queued.job_id);
    if (finalJob.status === "failed") {
      render3dResult("result-3d", finalJob);
      return;
    }
    const resultResponse = await fetch(`/api/v1/results/${queued.job_id}`);
    const resultPayload = await resultResponse.json();
    render3dResult("result-3d", resultPayload);
    renderImages("gallery-3d", [
      { label: "3D mask preview MIP", url: resultPayload.artifacts?.preview_mip },
    ]);
  } catch (error) {
    renderError("result-3d", error.message);
  } finally {
    setBusy(event.target, false);
  }
});
