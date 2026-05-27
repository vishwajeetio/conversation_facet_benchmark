const input = document.querySelector("#conversationInput");
const button = document.querySelector("#evaluateButton");
const rows = document.querySelector("#scoreRows");
const statusBox = document.querySelector("#status");
const summary = document.querySelector("#summary");
const facetCount = document.querySelector("#facetCount");
const facetLimit = document.querySelector("#facetLimit");
const conversationId = document.querySelector("#conversationId");
const activeRunKey = "oceanAcrossActiveRunId";

const example = [
  {
    speaker: "user",
    text: "I am worried about the deployment, but I checked the logs and can explain the tradeoffs clearly."
  },
  {
    speaker: "assistant",
    text: "Let's slow down, verify the risky step, and choose the safest reversible path."
  }
];

input.value = JSON.stringify(example, null, 2);

async function loadFacets() {
  const response = await fetch("/api/facets");
  const data = await response.json();
  facetCount.textContent = `${data.count} facets`;
}

function setStatus(message, isWarning = false) {
  statusBox.textContent = message;
  statusBox.style.color = isWarning ? "var(--warn)" : "var(--muted)";
}

function renderScores(data) {
  rows.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const result of data.results) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${result.turn_index}</td>
      <td>${result.facet_id}<br>${escapeHtml(result.facet_name)}</td>
      <td><span class="score">${result.score}</span></td>
      <td>${Math.round(result.confidence * 100)}%</td>
      <td>${escapeHtml(result.rationale)}</td>
    `;
    fragment.appendChild(row);
  }
  rows.appendChild(fragment);
  summary.textContent = `${data.results.length} scores from ${data.model}`;
  setStatus(`Scored ${data.turn_count} turns across ${data.facet_count} facets.`);
}

async function pollRun(runId) {
  localStorage.setItem(activeRunKey, runId);
  button.disabled = true;
  summary.textContent = "Running";
  setStatus(`Evaluation is running. Run id: ${runId}`);

  try {
    while (true) {
      const response = await fetch(`/api/evaluate/${encodeURIComponent(runId)}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not read run status");
      }

      if (data.status === "completed" && data.result) {
        renderScores(data.result);
        localStorage.removeItem(activeRunKey);
        return;
      }

      if (data.status === "failed") {
        throw new Error(data.message || "Evaluation failed");
      }

      setStatus(data.message || `Evaluation is running. Run id: ${runId}`);
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  } catch (error) {
    summary.textContent = "Needs attention";
    setStatus(error.message, true);
    localStorage.removeItem(activeRunKey);
  } finally {
    button.disabled = false;
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

button.addEventListener("click", async () => {
  rows.innerHTML = "";
  button.disabled = true;
  setStatus("Evaluating with the configured backend...");
  try {
    const turns = JSON.parse(input.value);
    const facetResponse = await fetch("/api/facets");
    const facetData = await facetResponse.json();
    const limit = Math.max(1, Number(facetLimit.value || 25));
    const facet_ids = facetData.facets.slice(0, limit).map((facet) => facet.facet_id);
    const response = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: conversationId.value || "ad-hoc",
        turns,
        facet_ids
      })
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Evaluation failed");
    }
    await pollRun(data.run_id);
  } catch (error) {
    summary.textContent = "Needs attention";
    setStatus(error.message, true);
  } finally {
    button.disabled = false;
  }
});

loadFacets().catch(() => setStatus("Could not load facets.", true));

const activeRunId = localStorage.getItem(activeRunKey);
if (activeRunId) {
  pollRun(activeRunId);
}
