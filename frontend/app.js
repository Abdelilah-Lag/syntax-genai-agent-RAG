"use strict";

const ROUTES = new Set(["research", "performance", "architecture", "insights"]);
const EXAMPLES = [
  "What are the main contraindications for warfarin?",
  "Compare the contraindications of warfarin and apixaban. What conditions are shared and which are unique?",
  "A patient currently on warfarin is prescribed sertraline. What is the bleeding risk and recommended monitoring?",
  "A patient with an eGFR of 25 mL/min is being considered for metformin. Is it contraindicated?",
  "What does INR mean?"
];
const TOOL_ORDER = ["search", "extract", "compare", "interaction", "glossary", "verify"];
const TOOL_DESCRIPTIONS = {
  search: "Mandatory hybrid search over the local corpus.",
  extract: "Reads document sections relevant to one drug and topic.",
  compare: "Retrieves the same aspect for multiple drugs.",
  interaction: "Finds label passages about a drug pair or class.",
  glossary: "Looks up pharmacology terms in the local glossary.",
  verify: "Retrieves evidence for a user-stated claim."
};
const STAGES = ["cache", "plan", "execute", "review", "revise", "synthesize", "finalize"];
const ARCHITECTURE_DETAILS = {
  input: "The Research page sends the user's question to POST /ask. Each request is independent; browser history is not sent as context.",
  api: "FastAPI validates the request, runs the agent, and returns the answer, evidence, citations, critic result and execution metadata.",
  orchestrator: "The runtime builds and reuses the agent. Its explicit graph connects planning, tool execution, synthesis, review, optional revision and finalization.",
  planner: "The planner asks for a JSON plan, then validates it against the corpus drug catalog and tool rules. A heuristic fallback is used if planning fails.",
  toolselection: "The validated plan selects tools for each sub-question. Search is always added; optional tools are planner-selected.",
  tools: "The six tools are search, extract, compare, interaction, glossary and verify. Independent tool jobs can run concurrently where applicable.",
  corpus: "The local corpus contains DailyMed drug labels, MedlinePlus monographs and a pharmacology glossary. Its documents are chunked and indexed at startup.",
  retrieval: "Search combines dense cosine rankings when embeddings are available with BM25 lexical rankings through Reciprocal Rank Fusion (RRF). If embeddings are unavailable, retrieval continues in BM25-only mode.",
  evidence: "Tool passages are deduplicated in EvidenceStore, prioritized within a character budget and assigned per-answer E1…En identifiers before generation.",
  generation: "The answer writer receives the question, plan, tool notes and numbered evidence; factual claims are instructed to cite that evidence.",
  quality: "Deterministic citation/number checks and a Critic LLM review the draft. Review availability and grounding scores are reported separately.",
  revision: "A revise verdict may trigger one bounded rewrite, followed by another Critic review. The graph does not allow an unbounded revision loop.",
  final: "Finalization resolves evidence IDs to document and section citations and returns the answer, cited sources, evidence metadata and execution trace.",
  response: "The API response carries the answer and real run metadata. The Research page renders its citations, retrieved passages and expandable tool trace."
};
const STRENGTHS = [
  ["Modular tool architecture", "Six separate tools have a shared contract and registry.", "Keeps retrieval actions independently testable and planner-selectable."],
  ["Validated JSON planner", "The planner output is normalized against corpus drugs and allowed tool rules.", "Prevents malformed plans and invalid tool arguments from flowing unchecked."],
  ["Mandatory search", "A search job is added for every plan step, even when no optional tool is selected.", "Preserves a baseline evidence path for every planned sub-question."],
  ["Parallel orchestration", "Independent tool jobs are submitted to a ThreadPoolExecutor.", "Reduces wall-clock time when a plan calls several independent tools."],
  ["Hybrid retrieval + RRF", "Dense scores and BM25 ranks are fused using Reciprocal Rank Fusion.", "Combines semantic matching with exact lexical signals."],
  ["Central EvidenceStore", "Passages are deduplicated and assigned stable E1…En identifiers.", "Provides a consistent evidence reference layer for writing and review."],
  ["Evidence-grounded generation", "The answer writer receives retrieved passages and is instructed to cite factual claims.", "Makes source attribution inspectable instead of returning a free-form answer."],
  ["Critic quality gate", "The Critic reviews claims while deterministic checks inspect citations and numbers.", "Adds a separate review stage and explicit findings to the response."],
  ["Bounded revision", "The graph allows at most one answer revision after a review.", "Limits runaway loops and bounds repeated generation work."],
  ["Citation and source tracking", "Evidence IDs resolve to document/section references and cited source records.", "Lets users inspect the documents used for each answer."],
  ["Execution observability", "The API exposes per-stage timings, tool traces and process-local runtime metrics.", "Helps identify slow stages and see which tools actually ran."],
  ["Offline evaluation", "A local 40-question dataset and fake-LLM tests support repeatable checks.", "Enables regression testing without making every test call a live model."]
];
const LIMITATIONS = [
  ["Corpus breadth is limited", "Verified: the current corpus contains ten supported medicines plus a glossary. Coverage outside it is not guaranteed."],
  ["Critic latency can dominate", "Verified in the supplied run report: the Critic and revision stages account for a large portion of several slow requests."],
  ["Evaluation breadth", "Verified: the public evaluation set contains 40 questions. This is a useful regression set, not broad clinical validation."],
  ["Tool selection is model-dependent", "Needs ongoing testing: a valid plan can still choose an unnecessary optional tool for a particular query."],
  ["Grounding is not medical correctness", "A high claim-grounding score only reflects the reviewer's evidence assessment; it is not clinical validation."],
  ["Citations can be inaccurate", "A citation can exist without fully supporting the attached claim. Citation presence alone is not proof of relevance."],
  ["Retrieval quality varies", "The report's source recall and precision show retrieval is measurable, but relevance and over-retrieval need continued tuning."],
  ["Model output can hallucinate", "Prompts, citations and a Critic reduce risk but do not guarantee that every generated statement is correct."]
];
const ROADMAP = {
  high: ["Improve retrieval precision and recall", "Tune planner decisions and avoid unnecessary tools", "Reduce unnecessary LLM calls", "Optimize Critic and revision latency", "Expand and diversify the evaluation set", "Strengthen citation-to-claim validation"],
  medium: ["Evaluate a reranking stage", "Add query rewriting when it improves retrieval", "Extend semantic caching carefully", "Improve retrieval diagnostics", "Add more detailed tracing"],
  future: ["Human-in-the-loop evaluation", "Broader independent evaluation benchmarks", "Assess multilingual support", "Advanced monitoring and alerting", "Production-grade access control and security"]
};
const METHODOLOGY = [
  ["Requirements", "Define the research task, safety boundaries and expected answer behavior."],
  ["Corpus preparation", "Inspect and normalize drug labels, patient monographs and glossary content."],
  ["Hybrid retrieval", "Build section-aware chunks and combine dense retrieval with BM25/RRF."],
  ["Planner & tools", "Add validated plan generation and specialized retrieval tools."],
  ["Grounded generation", "Constrain the answer writer to numbered retrieved evidence and citations."],
  ["Quality control", "Introduce deterministic checks, Critic review and a bounded revision path."],
  ["Evaluation", "Measure execution, retrieval, answer status, grounding and latency on a local benchmark."],
  ["Interface & telemetry", "Expose research traces, evidence inspection and process-level metrics."]
];
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[char]));
let healthSnapshot = null;
let metricsSnapshot = null;
let latest = null;
let sessionHistory = [];
let metricsLoading = false;

function currentRoute() {
  const route = location.hash.replace(/^#\/?/, "").split(/[?&]/)[0];
  return ROUTES.has(route) ? route : "research";
}

function navigate(path, replace = false) {
  const route = ROUTES.has(path) ? path : "research";
  const url = `#/${route}`;
  if (location.hash !== url) {
    history[replace ? "replaceState" : "pushState"]({ route }, "", url);
  }
  renderRoute(route);
}

function renderRoute(route = currentRoute()) {
  document.querySelectorAll("[data-page]").forEach(page => {
    page.hidden = page.dataset.page !== route;
  });
  document.querySelectorAll("[data-route-link]").forEach(link => {
    if (link.dataset.routeLink === route) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  const titles = {
    research: "Research Workspace",
    performance: "Execution & Performance",
    architecture: "System Architecture",
    insights: "Engineering Insights"
  };
  document.title = `${titles[route]} · GenAI Agent`;
  window.scrollTo({ top: 0, behavior: "instant" });
  if (route === "performance") refreshMetrics();
  if (route === "research" && latest) renderAnswer(latest, false);
}

function formatDuration(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Not available";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

function formatDate(value) {
  if (!value) return "Not available";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Not available" : date.toLocaleString();
}

function renderHealth(data) {
  healthSnapshot = data;
  const status = $("system-status");
  const currentPage = currentRoute();
  if (data.api_contract_version !== 4) {
    status.dataset.state = "error";
    status.querySelector("span").textContent = `API v${data.api_contract_version ?? "?"} · update required`;
  } else if (data.index_error) {
    status.dataset.state = "error";
    status.querySelector("span").textContent = "Index failed";
  } else if (data.index_ready) {
    status.dataset.state = "ready";
    status.querySelector("span").textContent = "API ready";
  } else {
    status.dataset.state = "checking";
    status.querySelector("span").textContent = "Index loading";
  }
  $("model-label").textContent = data.critic_model
    ? `Chat: ${data.chat_model || "Not available"} · Critic: ${data.critic_model}`
    : `Model: ${data.chat_model || "Not available"}`;
  $("corpus-status").textContent = data.index_error
    ? `Corpus error: ${data.index_error}`
    : data.index_ready ? "Corpus indexed · API ready" : "API reachable · indexing corpus…";
  $("ask-button").disabled = !data.index_ready || data.api_contract_version !== 4;
  $("stack-model").textContent = [
    `Chat: ${data.chat_model || "Not available"}`,
    `Fast: ${data.fast_model || "Not configured"}`,
    `Critic: ${data.critic_model || "Not available"}`
  ].join(" · ");
  if (currentPage === "research" && data.index_error) setRequestMessage(`Corpus initialization failed: ${data.index_error}`, true);
}

async function refreshHealth() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    if (!response.ok) throw new Error(`health endpoint returned ${response.status}`);
    renderHealth(await response.json());
  } catch (error) {
    healthSnapshot = null;
    $("system-status").dataset.state = "error";
    $("system-status").querySelector("span").textContent = "API unavailable";
    $("model-label").textContent = "Model details unavailable";
    $("corpus-status").textContent = "Cannot reach the API.";
    $("ask-button").disabled = true;
    $("stack-model").textContent = `API health data unavailable: ${error.message}`;
  }
}

function setRequestMessage(message, isError = false, loading = false) {
  const element = $("request-status");
  element.classList.toggle("error", isError);
  element.innerHTML = loading
    ? `<span class="spinner" aria-hidden="true"></span>${escapeHtml(message)}`
    : escapeHtml(message);
}

function renderInline(text) {
  let html = escapeHtml(text);
  html = html.replace(/\[([^\[\]]+\.md[^\[\]]*)\]/g, (_, citation) => {
    const filename = citation.split(/\s+§/)[0];
    return `<a class="citation-link" href="#evidence-viewer" data-source-file="${escapeHtml(filename)}" title="Open supporting evidence">${escapeHtml(citation)}</a>`;
  });
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  return html;
}

function renderMarkdown(value) {
  const blocks = String(value || "").trim().split(/\n{2,}/).filter(Boolean);
  return blocks.map(block => {
    const lines = block.split("\n");
    const heading = lines.length === 1 ? lines[0].match(/^(#{1,3})\s+(.+)$/) : null;
    if (heading) return `<h${heading[1].length}>${renderInline(heading[2])}</h${heading[1].length}>`;
    const isUnordered = lines.every(line => /^\s*[-*]\s+/.test(line));
    const isOrdered = lines.every(line => /^\s*\d+[.)]\s+/.test(line));
    if (isUnordered || isOrdered) {
      const tag = isOrdered ? "ol" : "ul";
      const pattern = isOrdered ? /^\s*\d+[.)]\s+/ : /^\s*[-*]\s+/;
      return `<${tag}>${lines.map(line => `<li>${renderInline(line.replace(pattern, ""))}</li>`).join("")}</${tag}>`;
    }
    return `<p>${lines.map(renderInline).join("<br>")}</p>`;
  }).join("");
}

function answerTextWithoutBoilerplate(data) {
  let answer = String(data.answer || "");
  const disclaimer = String(data.disclaimer || "").trim();
  if (disclaimer) answer = answer.replace(disclaimer, "").trim();
  answer = answer.replace(/\n*\*\*Sources:\*\*[^\n]*/g, "").trim();
  return answer;
}

function renderPlan(plan) {
  $("plan-origin").textContent = plan?.origin || "Not available";
  const steps = Array.isArray(plan?.steps) ? plan.steps : [];
  $("plan-details").innerHTML = steps.length ? steps.map(step => `
    <div class="plan-step">
      <strong>${escapeHtml(step.id)}. ${escapeHtml(step.sub_question)}</strong>
      <p>${step.drugs?.length ? `Drugs: ${escapeHtml(step.drugs.join(", "))}` : "No drug entity identified"}</p>
      ${(step.tools || []).map(tool => `<span class="tool-chip" title="${escapeHtml(step.tool_reasons?.[tool] || TOOL_DESCRIPTIONS[tool] || "")}">${escapeHtml(tool)}</span>`).join("")}
      ${Object.entries(step.tool_reasons || {}).map(([tool, reason]) => `<p>${escapeHtml(tool)}: ${escapeHtml(reason)}</p>`).join("")}
    </div>`).join("") : `<div class="empty-state">No planner steps were returned for this request.</div>`;
}

function renderCritic(data) {
  const critic = data.critic || {};
  const reviewStatus = critic.review_status || (critic.llm_ok === false ? "unavailable" : "Not available");
  const verdict = critic.verdict || "Not available";
  const grounding = typeof critic.grounding_score === "number" && Number.isFinite(critic.grounding_score)
    ? `${Math.round(critic.grounding_score * 100)}%`
    : "Not available";
  const verdictElement = $("critic-verdict");
  verdictElement.textContent = verdict.replaceAll("_", " ");
  verdictElement.dataset.status = reviewStatus === "completed" ? verdict : reviewStatus;
  const flags = Array.isArray(critic.flags) ? critic.flags : [];
  $("critic-metrics").innerHTML = [
    ["Review status", reviewStatus.replaceAll("_", " ")],
    ["Grounding score", grounding],
    ["Claims checked", Array.isArray(critic.claims) ? critic.claims.length : "Not available"],
    ["Attempts / revisions", `${critic.attempts ?? "Not available"} / ${critic.revisions ?? "Not available"}`]
  ].map(([label, value]) => `<div class="critic-stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  const summary = String(critic.summary || "").trim();
  $("critic-summary").hidden = !summary;
  $("critic-summary").textContent = summary;
  $("critic-flags").innerHTML = flags.length ? flags.map(flag => `
    <div class="flag-item"><strong>${escapeHtml(String(flag.type || "Review note").replaceAll("_", " "))}</strong>
      ${flag.claim ? `<span>${escapeHtml(flag.claim)}</span>` : ""}
      ${flag.note ? `<div class="muted small">${escapeHtml(flag.note)}</div>` : ""}
    </div>`).join("") : `<div class="empty-state">${reviewStatus === "unavailable" ? "The answer was not validated by the Critic." : reviewStatus === "not_required" ? "No answer claims required Critic review." : "No review flags were returned."}</div>`;
}

function renderSources(data) {
  const sources = Array.isArray(data.sources) ? data.sources : [];
  $("source-count").textContent = `${sources.length} ${sources.length === 1 ? "source" : "sources"}`;
  $("source-list").innerHTML = sources.length ? sources.map(source => `
    <article class="source-card" data-source-card="${escapeHtml(source.file)}">
      <div class="source-card-head"><div><strong>${escapeHtml(source.file || "Source document")}</strong><small>${escapeHtml(source.source_type || "Source")} · ${escapeHtml(source.section || "Section not available")}</small></div><span class="status-pill" data-status="supported">Cited</span></div>
      ${source.snippet ? `<p>${escapeHtml(source.snippet)}</p>` : ""}
      ${source.url ? `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">Open original source ↗</a>` : ""}
    </article>`).join("") : `<div class="empty-state">No cited sources were returned for this answer.</div>`;
}

function renderEvidence(data) {
  const items = Array.isArray(data.evidence) ? data.evidence : [];
  const filter = ($("evidence-filter").value || "").trim().toLowerCase();
  const filtered = items.filter(item => `${item.evidence_id} ${item.file} ${item.section}`.toLowerCase().includes(filter));
  $("evidence-count").textContent = `${items.length} ${items.length === 1 ? "passage" : "passages"}`;
  $("evidence-list").innerHTML = filtered.length ? filtered.map(item => `
    <details class="evidence-item" data-evidence-card data-source-name="${escapeHtml(item.file)}" id="evidence-${escapeHtml(item.evidence_id)}">
      <summary><span><strong>[${escapeHtml(item.evidence_id)}] ${escapeHtml(item.file)}</strong><span class="evidence-meta">${escapeHtml(item.source_type)} · ${escapeHtml(item.section)} · score ${typeof item.score === "number" ? item.score.toFixed(3) : "Not available"} · ${(item.tools || []).map(escapeHtml).join(", ")}</span></span><span class="status-pill" data-status="${item.cited ? "supported" : "partial"}">${item.cited ? "Cited" : "Retrieved"}</span></summary>
      <blockquote class="evidence-snippet">${escapeHtml(item.snippet || "Passage text not available.")}</blockquote>
    </details>`).join("") : `<div class="empty-state">${items.length ? "No retrieved passages match this filter." : "The API returned no retrieved evidence passages."}</div>`;
}

function renderTrace(data) {
  const timings = data.timings_ms && typeof data.timings_ms === "object" ? data.timings_ms : {};
  const calls = Array.isArray(data.tool_trace) ? data.tool_trace : [];
  const plan = data.plan || {};
  const stages = Object.entries(timings);
  const reviews = Array.isArray(data.critic?.reviews) ? data.critic.reviews : [];
  $("trace-summary").textContent = `${calls.length} tool calls · ${stages.length} timed stages`;
  const metadata = [
    ["Run ID", data.request_id || "Not supplied"],
    ["Executed at", formatDate(data.timestamp)],
    ["Answer status", data.answer_status || "Not available"],
    ["Tools executed", Array.isArray(data.tools_used) && data.tools_used.length ? [...new Set(data.tools_used)].join(", ") : "No tool calls reported"]
  ].map(([label, value]) => `<span><strong>${escapeHtml(label)}:</strong> ${escapeHtml(value)}</span>`).join("");
  const toolTable = calls.length ? `<div class="table-wrap"><table class="trace-table"><thead><tr><th>Plan step</th><th>Tool</th><th>Passages</th><th>Duration</th><th>Status</th></tr></thead><tbody>${calls.map(call => `
    <tr><td>${escapeHtml(call.step ?? "Not available")}</td><td>${escapeHtml(call.tool || "Not available")}</td><td>${escapeHtml(call.passages ?? "Not available")}</td><td>${formatDuration(typeof call.ms === "number" ? call.ms : null)}</td><td><span class="status-pill" data-status="${call.error ? "error" : "ready"}">${call.error ? `Failed: ${escapeHtml(call.error)}` : "Success"}</span></td></tr>`).join("")}</tbody></table></div><p class="chart-note">Calls in a plan step may have run concurrently; this API does not expose per-call start/end timestamps.</p>` : `<div class="empty-state">No tool execution records were returned.</div>`;
  const timingList = stages.length ? `<div class="timing-list">${stages.map(([stage, duration]) => `<div class="timing-item"><span>${escapeHtml(stage)}</span><strong>${formatDuration(typeof duration === "number" ? duration : null)}</strong></div>`).join("")}</div>` : `<div class="empty-state">Stage timing data was not returned.</div>`;
  const selectedTools = Array.isArray(plan.steps) ? plan.steps.flatMap(step => (step.tools || []).map(tool => `${tool} (step ${step.id})`)) : [];
  const reviewList = reviews.length ? `<ul>${reviews.map(review => `<li>Attempt ${escapeHtml(review.attempt)}: ${escapeHtml(review.verdict)} · ${formatDuration(typeof review.duration_ms === "number" ? review.duration_ms : null)}</li>`).join("")}</ul>` : `<span class="muted small">No per-review timing details returned.</span>`;
  $("trace-content").innerHTML = `
    <div class="trace-metadata">${metadata}</div>
    <section class="trace-block"><h4>Planner decisions</h4>${Array.isArray(plan.steps) && plan.steps.length ? `<ul>${plan.steps.map(step => `<li>Step ${escapeHtml(step.id)}: ${escapeHtml(step.sub_question)} · selected tools: ${escapeHtml((step.tools || []).join(", ") || "search only")}</li>`).join("")}</ul>` : `<div class="empty-state">No plan steps were returned. Planner origin: ${escapeHtml(plan.origin || "Not available")}.</div>`}${selectedTools.length ? `<p class="muted small">Planned tool calls: ${escapeHtml(selectedTools.join(" · "))}</p>` : ""}</section>
    <section class="trace-block"><h4>Executed tool calls</h4>${toolTable}</section>
    <section class="trace-block"><h4>Stage timings</h4>${timingList}</section>
    <section class="trace-block"><h4>Critic reviews</h4>${reviewList}</section>`;
}

function renderAnswer(data, scroll = true) {
  latest = data;
  $("run-panel").hidden = false;
  $("run-title").textContent = data.question || "Research response";
  $("run-meta").textContent = [
    data.timestamp ? formatDate(data.timestamp) : "",
    typeof data.latency_ms === "number" ? formatDuration(data.latency_ms) : "",
    data.cache_hit ? "API answer cache hit" : "Fresh API execution"
  ].filter(Boolean).join(" · ") || "Run metadata not available";
  const status = $("answer-status");
  status.textContent = data.answer_status || "Not available";
  status.dataset.status = data.answer_status || "unknown";
  $("citation-count").textContent = `${Array.isArray(data.sources) ? data.sources.length : 0} cited sources`;
  $("answer-body").innerHTML = renderMarkdown(answerTextWithoutBoilerplate(data));
  renderPlan(data.plan);
  renderCritic(data);
  renderSources(data);
  renderEvidence(data);
  renderTrace(data);
  if (scroll) $("run-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderHistory() {
  $("conversation").innerHTML = sessionHistory.length ? sessionHistory.map((item, index) => `
    <div class="history-row"><button type="button" data-history-index="${index}" title="${escapeHtml(item.question)}">${escapeHtml(item.question)}</button><small>${formatDate(item.timestamp)} · ${escapeHtml(item.status)}</small></div>`).join("")
    : `<div class="empty-state">Your recent queries appear here. Each request is independent; previous questions are not sent as context.</div>`;
}

function setLoading(loading) {
  $("ask-button").disabled = loading || !healthSnapshot?.index_ready || healthSnapshot.api_contract_version !== 4;
  $("clear-question").disabled = loading;
  if (loading) setRequestMessage("Planning, retrieving and reviewing with the agent…", false, true);
}

async function ask(event) {
  event?.preventDefault();
  const question = $("question").value.trim();
  if (question.length < 3) {
    setRequestMessage("Enter a question with at least 3 characters.", true);
    return;
  }
  if (!healthSnapshot?.index_ready || healthSnapshot.api_contract_version !== 4) {
    setRequestMessage("The API or corpus index is not ready yet. Check the service status above.", true);
    return;
  }
  setLoading(true);
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `Request failed (${response.status})`);
    renderAnswer(payload);
    sessionHistory.unshift({
      question: payload.question || question,
      timestamp: payload.timestamp || new Date().toISOString(),
      status: payload.answer_status || "not available",
      response: payload
    });
    sessionHistory = sessionHistory.slice(0, 12);
    renderHistory();
    setRequestMessage("Research response received from the API.");
    await refreshMetrics(false);
  } catch (error) {
    setRequestMessage(error instanceof Error ? error.message : "The request failed.", true);
  } finally {
    setLoading(false);
  }
}

function statCard(label, value, note) {
  return `<article class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></article>`;
}

function metricGroup(title, description, cards) {
  return `<section class="metric-group"><div class="metric-group-heading"><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p></div><div class="metric-grid">${cards}</div></section>`;
}

function renderDistribution(container, entries, total) {
  if (!entries.length || total <= 0) {
    container.innerHTML = `<div class="empty-state">No observed data yet.</div>`;
    return;
  }
  container.innerHTML = entries.map(([label, count]) => {
    const width = Math.max(0, Math.min(100, total ? count / total * 100 : 0));
    return `<div class="distribution-row"><span>${escapeHtml(String(label).replaceAll("_", " "))}</span><div class="bar-track"><div class="bar-fill" style="width:${width.toFixed(1)}%"></div></div><strong>${escapeHtml(count)}</strong></div>`;
  }).join("");
}

function renderMetrics(metrics) {
  metricsSnapshot = metrics;
  const total = Number(metrics.total_requests) || 0;
  const history = Array.isArray(metrics.history) ? metrics.history : [];
  const availability = Number.isFinite(metrics.critic_review_available_rate)
    ? `${Math.round(metrics.critic_review_available_rate * 100)}%`
    : "Not available";
  const grounding = Number.isFinite(metrics.average_grounding_score)
    ? `${Math.round(metrics.average_grounding_score * 100)}%`
    : "Not available";
  const toolAverage = Number.isFinite(metrics.average_tools_per_request)
    ? String(metrics.average_tools_per_request)
    : "Not available";
  $("metrics-scope").textContent = `${metrics.scope || "Telemetry scope not returned"}${metrics.history_limit ? ` · retained records: up to ${metrics.history_limit}` : ""}`;
  $("analytics-summary").innerHTML = [
    metricGroup("Latency & API reliability", "Observed at the API boundary for this process.", [
      statCard("Observed requests", total, "Current process history"),
      statCard("Successful requests", metrics.successful_requests ?? "Not available", "API requests recorded as successful"),
      statCard("Failed requests", metrics.failed_requests ?? "Not available", "API requests recorded as failed"),
      statCard("Average response time", formatDuration(metrics.average_latency_ms), "API-boundary latency; not a model benchmark")
    ].join("")),
    metricGroup("Answer quality & review", "Only values actually returned by the answer and Critic pipeline are shown.", [
      statCard("Mean grounding score", grounding, "Only numeric scores returned by completed reviews"),
      statCard("Critic review availability", availability, "Not-required reviews excluded by backend telemetry"),
      statCard("Average revisions", Number.isFinite(metrics.average_revisions) ? String(metrics.average_revisions) : "Not available", "Observed answer revisions"),
      statCard("Average Critic attempts", Number.isFinite(metrics.average_critic_attempts) ? String(metrics.average_critic_attempts) : "Not available", "Attempts reported in runtime records")
    ].join("")),
    metricGroup("Tool execution", "Aggregated from actual tool calls recorded by the running API.", [
      statCard("Average tools per request", toolAverage, "Actual tool-call records")
    ].join("")),
    `<section class="metric-group metric-coverage"><div class="metric-group-heading"><h3>Retrieval &amp; planner evaluation</h3><p>These quality metrics are not exposed by the live /metrics endpoint; they are not inferred from request latency or tool counts.</p></div><div class="metric-coverage-grid"><article><strong>Retrieval quality</strong><span>Source recall and precision are reported by the offline evaluation runner.</span></article><article><strong>Planner quality</strong><span>Plan validity, fallback rate and planned-tool execution are reported by offline evaluation.</span></article></div></section>`
  ].join("");

  const toolMap = metrics.tools || {};
  const registered = Array.isArray(metrics.registered_tools) ? metrics.registered_tools : TOOL_ORDER;
  $("tool-performance").innerHTML = registered.length ? registered.map(name => {
    const item = toolMap[name];
    return item ? `<tr><td title="${escapeHtml(TOOL_DESCRIPTIONS[name] || "")}">${escapeHtml(name)}</td><td>${escapeHtml(item.calls ?? "Not available")}</td><td>${formatDuration(item.average_ms)}</td><td>${formatDuration(item.min_ms)} / ${formatDuration(item.max_ms)}</td><td>${Number.isFinite(item.success_rate) ? `${Math.round(item.success_rate * 100)}%` : "Not available"}</td></tr>`
      : `<tr><td title="${escapeHtml(TOOL_DESCRIPTIONS[name] || "")}">${escapeHtml(name)}</td><td colspan="4" class="muted">No observed calls in retained history</td></tr>`;
  }).join("") : `<tr><td colspan="5" class="muted">Tool registry data not available.</td></tr>`;

  const averages = metrics.average_component_ms || {};
  const observedStages = STAGES.filter(stage => Object.hasOwn(averages, stage));
  $("component-performance").innerHTML = STAGES.map(stage => {
    const value = averages[stage];
    if (!observedStages.includes(stage) || !Number.isFinite(value)) {
      return `<div class="bar-row"><span class="bar-label">${escapeHtml(stage)}</span><span class="bar-track"></span><span class="bar-value">Not observed</span></div>`;
    }
    const max = Math.max(1, ...observedStages.map(key => Number(averages[key]) || 0));
    const width = Math.max(1, Number(value) / max * 100);
    return `<div class="bar-row"><span class="bar-label">${escapeHtml(stage)}</span><span class="bar-track"><span class="bar-fill" style="width:${width.toFixed(1)}%"></span></span><span class="bar-value">${formatDuration(value)}</span></div>`;
  }).join("");

  $("history-count").textContent = `${history.length} of ${metrics.history_limit ?? "available"} retained observations`;
  const latency = history.slice(-40);
  const maxLatency = Math.max(1, ...latency.map(item => Number(item.latency_ms) || 0));
  $("latency-chart").innerHTML = latency.length ? latency.map((item, index) => {
    const duration = Number(item.latency_ms);
    const height = Number.isFinite(duration) ? Math.max(2, duration / maxLatency * 100) : 0;
    return `<div class="latency-column" title="${escapeHtml(formatDate(item.timestamp))} · ${formatDuration(duration)} · ${item.success ? "Success" : "Failed"}"><span class="latency-bar ${item.success ? "" : "failed"}" style="height:${height.toFixed(1)}%"></span><small>${index + 1}</small></div>`;
  }).join("") : `<div class="empty-state">No request history has been collected by this process.</div>`;

  const scored = history.filter(item => typeof item.critic?.grounding_score === "number" && Number.isFinite(item.critic.grounding_score));
  const maxScore = 1;
  $("grounding-chart").innerHTML = scored.length ? scored.slice(-40).map((item, index) => {
    const score = Math.max(0, Math.min(maxScore, item.critic.grounding_score));
    return `<div class="history-column" title="${escapeHtml(formatDate(item.timestamp))} · grounding ${Math.round(score * 100)}%"><span class="history-bar" style="height:${Math.max(2, score * 100).toFixed(1)}%"></span><small>${index + 1}</small></div>`;
  }).join("") : `<div class="empty-state">No numeric Critic grounding scores in retained history.</div>`;

  const reviewCounts = Object.entries(metrics.critic_review_statuses || {});
  renderDistribution($("review-distribution"), reviewCounts, reviewCounts.reduce((sum, [, count]) => sum + (Number(count) || 0), 0));
  renderDistribution($("request-distribution"), [
    ["successful", Number(metrics.successful_requests) || 0],
    ["failed", Number(metrics.failed_requests) || 0]
  ], total);
  const revisions = history.map(item => Number(item.critic?.revisions)).filter(Number.isFinite);
  if (revisions.length) {
    const revisionCounts = [...new Set(revisions)].sort((a, b) => a - b).map(value => [
      `${value} ${value === 1 ? "revision" : "revisions"}`,
      revisions.filter(count => count === value).length
    ]);
    renderDistribution($("revision-chart"), revisionCounts, revisions.length);
  } else {
    $("revision-chart").innerHTML = `<div class="empty-state">No revision history is available in this telemetry snapshot.</div>`;
  }
  $("metrics-refreshed").textContent = `Last refreshed ${new Date().toLocaleTimeString()}`;
}

async function refreshMetrics(showLoading = true) {
  if (metricsLoading) return;
  metricsLoading = true;
  $("metrics-error").hidden = true;
  if (showLoading && !metricsSnapshot) $("metrics-loading").hidden = false;
  const button = $("refresh-metrics");
  if (button) button.disabled = true;
  try {
    const response = await fetch("/metrics", { cache: "no-store" });
    if (!response.ok) throw new Error(`Metrics endpoint returned ${response.status}`);
    renderMetrics(await response.json());
  } catch (error) {
    $("metrics-error").textContent = `Runtime telemetry is unavailable: ${error instanceof Error ? error.message : "request failed"}.`;
    $("metrics-error").hidden = false;
    if (!metricsSnapshot) {
      $("analytics-summary").innerHTML = `<div class="empty-state">No telemetry snapshot is available.</div>`;
      $("tool-performance").innerHTML = `<tr><td colspan="5" class="muted">Tool metrics not available.</td></tr>`;
    }
  } finally {
    metricsLoading = false;
    $("metrics-loading").hidden = true;
    if (button) button.disabled = false;
  }
}

function renderArchitecture() {
  $("architecture-detail").textContent = ARCHITECTURE_DETAILS.input;
  document.querySelectorAll("[data-architecture]").forEach(button => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-architecture]").forEach(node => node.setAttribute("aria-pressed", String(node === button)));
      $("architecture-detail").textContent = ARCHITECTURE_DETAILS[button.dataset.architecture] || "Implementation details are not available.";
    });
  });
  document.querySelector('[data-architecture="input"]')?.setAttribute("aria-pressed", "true");
}

function renderInsights() {
  $("strengths-grid").innerHTML = STRENGTHS.map(([title, implementation, value]) => `
    <article class="insight-card"><h3>${escapeHtml(title)}</h3><p>${escapeHtml(implementation)}</p><small><strong>Engineering value:</strong> ${escapeHtml(value)}</small></article>`).join("");
  $("limitations-list").innerHTML = LIMITATIONS.map(([title, detail]) => `
    <div class="limitation-item"><span aria-hidden="true">!</span><div><strong>${escapeHtml(title)}</strong><small>${escapeHtml(detail)}</small></div></div>`).join("");
  $("roadmap-grid").innerHTML = [
    ["high", "High priority", ROADMAP.high],
    ["medium", "Medium priority", ROADMAP.medium],
    ["future", "Future", ROADMAP.future]
  ].map(([priority, label, items]) => `<section class="roadmap-column" data-priority="${priority}"><h3>${escapeHtml(label)}</h3><ul>${items.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul></section>`).join("");
  $("methodology-timeline").innerHTML = METHODOLOGY.map(([title, description]) => `<li><strong>${escapeHtml(title)}</strong><small>${escapeHtml(description)}</small></li>`).join("");
}

function setupEvents() {
  document.addEventListener("click", event => {
    const link = event.target.closest("[data-route-link]");
    if (!link) return;
    const route = link.dataset.routeLink;
    if (!ROUTES.has(route)) return;
    event.preventDefault();
    navigate(route);
  });
  window.addEventListener("popstate", () => renderRoute(currentRoute()));
  window.addEventListener("hashchange", () => renderRoute(currentRoute()));
  $("ask-form").addEventListener("submit", ask);
  $("question").addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      ask(event);
    }
  });
  $("examples").innerHTML = EXAMPLES.map((question, index) => `
    <button class="example-question" type="button" data-example="${index}">${escapeHtml(question)}</button>`).join("");
  $("examples").addEventListener("click", event => {
    const button = event.target.closest("[data-example]");
    if (!button) return;
    $("question").value = EXAMPLES[Number(button.dataset.example)] || "";
    $("question").focus();
    setRequestMessage("Example added. Submit it when ready.");
  });
  $("clear-question").addEventListener("click", () => {
    $("question").value = "";
    setRequestMessage("");
    $("question").focus();
  });
  $("clear-history").addEventListener("click", () => {
    sessionHistory = [];
    renderHistory();
  });
  $("copy-answer").addEventListener("click", async () => {
    if (!latest?.answer) return;
    try {
      await navigator.clipboard.writeText(latest.answer);
      $("copy-answer").textContent = "Copied";
      setTimeout(() => { $("copy-answer").textContent = "Copy answer"; }, 1400);
    } catch {
      setRequestMessage("Clipboard access is unavailable in this browser.", true);
    }
  });
  $("refresh-metrics").addEventListener("click", () => refreshMetrics());
  $("evidence-filter").addEventListener("input", () => latest && renderEvidence(latest));
  $("answer-body").addEventListener("click", event => {
    const citation = event.target.closest("[data-source-file]");
    if (!citation) return;
    event.preventDefault();
    const file = citation.dataset.sourceFile;
    const evidence = [...document.querySelectorAll("[data-evidence-card]")].find(item => item.dataset.sourceName === file);
    const source = [...document.querySelectorAll("[data-source-card]")].find(item => item.dataset.sourceCard === file);
    if (evidence) {
      evidence.open = true;
      evidence.scrollIntoView({ behavior: "smooth", block: "center" });
    } else if (source) {
      source.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
  $("conversation").addEventListener("click", event => {
    const button = event.target.closest("[data-history-index]");
    const item = button ? sessionHistory[Number(button.dataset.historyIndex)] : null;
    if (!item) return;
    $("question").value = item.question;
    if (item.response) renderAnswer(item.response);
  });
  $("theme-toggle").addEventListener("click", () => {
    const theme = document.body.dataset.theme === "dark" ? "light" : "dark";
    document.body.dataset.theme = theme;
    localStorage.setItem("syntax-rag-theme", theme);
  });
}

function setupLogoFallbacks() {
  document.querySelectorAll(".brand-logo, .syntax-logo").forEach(image => {
    image.addEventListener("error", () => {
      image.hidden = true;
    });
  });
}

function init() {
  const savedTheme = localStorage.getItem("syntax-rag-theme");
    if (savedTheme === "dark") document.body.dataset.theme = "dark";
  setupEvents();
  setupLogoFallbacks();
  renderArchitecture();
  renderInsights();
  renderHistory();
  refreshHealth();
  if (!ROUTES.has(location.hash.replace(/^#\/?/, "").split(/[?&]/)[0])) {
    navigate("research", true);
  } else {
    renderRoute(currentRoute());
  }
}

init();
