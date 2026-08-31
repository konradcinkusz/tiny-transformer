(() => {
  "use strict";

  const form = document.getElementById("encode-form");
  const statusEl = document.getElementById("status");
  const runButton = document.getElementById("run-button");
  const errorBox = document.getElementById("error-box");
  const resultsSection = document.getElementById("results");
  const configSummary = document.getElementById("config-summary");
  const tokensEl = document.getElementById("tokens");
  const attentionControls = document.getElementById("attention-controls");
  const attentionLayerSelect = document.getElementById("attention-layer");
  const attentionHeadSelect = document.getElementById("attention-head");
  const weightsTrainedRadio = document.getElementById("weights-trained");
  const textInputGroup = document.getElementById("text-input-group");
  const advancedSettings = document.getElementById("advanced-settings");
  const hintEmbeddings = document.getElementById("hint-embeddings");
  const hintTokens = document.getElementById("hint-tokens");

  const POS_COLOR = [91, 66, 243]; // indigo - positive values
  const NEG_COLOR = [211, 47, 47]; // red - negative values
  const MAX_BLEND = 0.75; // never fully saturate, so dark cell text stays legible

  // The last successful response, kept so the layer/head selectors can
  // re-render the attention heatmap from already-fetched data instead of
  // making a new request.
  let currentResult = null;

  form.addEventListener("submit", onSubmit);
  attentionLayerSelect.addEventListener("change", renderAttentionHeatmap);
  attentionHeadSelect.addEventListener("change", renderAttentionHeatmap);
  for (const radio of document.querySelectorAll('input[name="weights"]'))
    radio.addEventListener("change", updateWeightsChoiceUi);
  updateWeightsChoiceUi();

  // Text/advanced settings are meaningless once the trained model is
  // selected (it always runs on its own fixed demo task - see
  // EncodeRequest's comment on why), so grey them out rather than leaving
  // inputs on screen that the request will silently ignore.
  function updateWeightsChoiceUi() {
    const useTrainedModel = weightsTrainedRadio.checked;
    textInputGroup.classList.toggle("is-disabled", useTrainedModel);
    document.getElementById("text").disabled = useTrainedModel;
    advancedSettings.classList.toggle("is-disabled", useTrainedModel);
    for (const input of advancedSettings.querySelectorAll("input")) input.disabled = useTrainedModel;
  }

  async function onSubmit(event) {
    event.preventDefault();
    hideError();
    setBusy(true);

    const useTrainedModel = weightsTrainedRadio.checked;
    const payload = useTrainedModel
      ? { useTrainedModel: true }
      : {
          text: document.getElementById("text").value,
          dModel: numberOrNull("dModel"),
          dK: numberOrNull("dK"),
          ffHidden: numberOrNull("ffHidden"),
          numHeads: numberOrNull("numHeads"),
          numLayers: numberOrNull("numLayers"),
          seed: numberOrNull("seed"),
        };

    try {
      const response = await fetch("/api/encode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (response.status === 429) {
        const body = await safeJson(response);
        const retry = body && body.retryAfter ? ` Try again in ~${Math.ceil(body.retryAfter)}s.` : "";
        showError(`Rate limit reached.${retry}`);
        return;
      }

      if (!response.ok) {
        const body = await safeJson(response);
        showError(formatErrors(body));
        return;
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      showError(`Could not reach the API: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  function numberOrNull(id) {
    const raw = document.getElementById(id).value;
    if (raw === "" || raw === null) return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  }

  function formatErrors(body) {
    if (body && body.errors && typeof body.errors === "object") {
      return Object.entries(body.errors)
        .map(([field, messages]) => `${field}: ${Array.isArray(messages) ? messages.join(" ") : messages}`)
        .join(" — ");
    }
    if (body && body.title) return body.title;
    return "Something went wrong. Please try again.";
  }

  async function safeJson(response) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  function setBusy(busy) {
    runButton.disabled = busy;
    statusEl.textContent = busy ? "Running the encoder…" : "";
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function hideError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function renderResults(data) {
    resultsSection.hidden = false;
    currentResult = data;

    hintEmbeddings.textContent = data.config.usedTrainedModel
      ? "Learned lookup vector per token id, from the pretrained demo model - shape [tokens × d_model]."
      : "Random (untrained) lookup vector per token id - shape [tokens × d_model].";
    hintTokens.textContent = data.config.usedTrainedModel
      ? "The fixed synthetic token ids the trained model was trained on - not derived from any text (see the \"Weights\" note above)."
      : "Each distinct character seen in the input gets the next free token id, in order of first appearance.";

    renderConfigSummary(data.config);
    renderTokens(data.tokens, data.tokenIds);

    const tokenLabels = data.tokens.map((t, i) => `${i}:${t}`);
    const dimLabels = (n) => Array.from({ length: n }, (_, i) => `d${i}`);

    renderMatrix("mat-embeddings", data.embeddings, tokenLabels, dimLabels(data.config.dModel));
    renderMatrix("mat-positional", data.positionalEncoding, tokenLabels, dimLabels(data.config.dModel));
    populateAttentionSelectors(data.config.numLayers, data.config.numHeads);
    renderAttentionHeatmap();
    renderMatrix("mat-output", data.encoderOutput, tokenLabels, dimLabels(data.config.dModel));

    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderConfigSummary(config) {
    configSummary.replaceChildren();
    const entries = [
      ["weights", config.usedTrainedModel ? "trained (demo task)" : "random"],
      ["sequence length", config.sequenceLength],
      ["vocab size (this request)", config.vocabSize],
      ["d_model", config.dModel],
      ["d_k", config.dK],
      ["ff hidden", config.ffHidden],
      ["heads", config.numHeads],
      ["layers", config.numLayers],
    ];
    if (!config.usedTrainedModel) entries.push(["seed", config.seed]);
    for (const [label, value] of entries) {
      const span = document.createElement("span");
      span.textContent = `${label}: ${value}`;
      configSummary.appendChild(span);
    }
  }

  // Layer select defaults to the last layer and head select defaults to
  // "average across heads" - together that reproduces the single heatmap
  // this panel always showed before per-layer/per-head detail existed.
  // With one layer and one head (today's default request), the selector row
  // stays hidden so the panel looks exactly as before.
  function populateAttentionSelectors(numLayers, numHeads) {
    attentionLayerSelect.replaceChildren();
    for (let layer = 0; layer < numLayers; layer++) {
      const option = document.createElement("option");
      option.value = String(layer);
      option.textContent = `${layer}`;
      attentionLayerSelect.appendChild(option);
    }
    attentionLayerSelect.value = String(numLayers - 1);

    attentionHeadSelect.replaceChildren();
    const averageOption = document.createElement("option");
    averageOption.value = "average";
    averageOption.textContent = "average";
    attentionHeadSelect.appendChild(averageOption);
    for (let head = 0; head < numHeads; head++) {
      const option = document.createElement("option");
      option.value = String(head);
      option.textContent = `${head}`;
      attentionHeadSelect.appendChild(option);
    }
    attentionHeadSelect.value = "average";

    attentionControls.hidden = numLayers <= 1 && numHeads <= 1;
  }

  function renderAttentionHeatmap() {
    if (!currentResult) return;

    const layer = Number(attentionLayerSelect.value);
    const headSelection = attentionHeadSelect.value;
    const perHead = currentResult.attentionWeightsPerLayer[layer];
    const matrix = headSelection === "average" ? averageMatrices(perHead) : perHead[Number(headSelection)];

    const tokenLabels = currentResult.tokens.map((t, i) => `${i}:${t}`);
    renderMatrix("mat-attention", matrix, tokenLabels, tokenLabels);
  }

  function averageMatrices(matrices) {
    const rows = matrices[0].length;
    const cols = matrices[0][0].length;
    const result = Array.from({ length: rows }, () => new Array(cols).fill(0));

    for (const matrix of matrices)
      for (let i = 0; i < rows; i++)
        for (let j = 0; j < cols; j++)
          result[i][j] += matrix[i][j];

    const inv = 1 / matrices.length;
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++)
        result[i][j] *= inv;

    return result;
  }

  function renderTokens(tokens, tokenIds) {
    tokensEl.replaceChildren();
    tokens.forEach((token, i) => {
      const chip = document.createElement("span");
      chip.className = "token-chip";

      const text = document.createTextNode(token);
      const id = document.createElement("span");
      id.className = "id";
      id.textContent = ` #${tokenIds[i]}`;

      chip.appendChild(text);
      chip.appendChild(id);
      tokensEl.appendChild(chip);
    });
  }

  function renderMatrix(containerId, matrix, rowLabels, colLabels) {
    const container = document.getElementById(containerId);
    container.replaceChildren();

    if (!matrix || matrix.length === 0) return;

    let maxAbs = 0;
    for (const row of matrix) for (const v of row) maxAbs = Math.max(maxAbs, Math.abs(v));
    if (maxAbs === 0) maxAbs = 1;

    const table = document.createElement("table");
    table.className = "heatmap";

    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    headRow.appendChild(document.createElement("th"));
    colLabels.forEach((label) => {
      const th = document.createElement("th");
      th.textContent = label;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    matrix.forEach((row, i) => {
      const tr = document.createElement("tr");

      const rowHeader = document.createElement("th");
      rowHeader.className = "row-label";
      rowHeader.textContent = rowLabels[i] ?? String(i);
      tr.appendChild(rowHeader);

      row.forEach((value) => {
        const td = document.createElement("td");
        td.textContent = value.toFixed(3);
        td.style.backgroundColor = cellColor(value, maxAbs);
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    container.appendChild(table);
  }

  function cellColor(value, maxAbs) {
    const t = Math.max(-1, Math.min(1, value / maxAbs));
    const color = t >= 0 ? POS_COLOR : NEG_COLOR;
    const alpha = Math.abs(t) * MAX_BLEND;
    const [r, g, b] = color.map((channel) => Math.round(255 + (channel - 255) * alpha));
    return `rgb(${r}, ${g}, ${b})`;
  }
})();
